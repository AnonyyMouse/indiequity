"""
Rigorous Discounted Cash Flow (DCF) and equity valuation engine for Indian markets.

Methodology & Standards:
1. Free Cash Flow to Firm (FCFF):
   FCFF = EBIT * (1 - t) + D&A - CapEx - Change_in_Operating_NWC
   Operating working capital strictly excludes cash, debt, and financing liabilities.
   Negative FCFF is preserved and not replaced with arbitrary positive proxies.
2. Explicit Multi-Year Financial Forecasts:
   Detailed line-by-line forecasts (Revenue, Operating Margin, EBIT, Taxes, D&A,
   CapEx, ΔNWC, Reinvestment, ROIC) for 5-year or 10-year transition periods.
3. Market-Value WACC Estimation:
   CAPM Cost of Equity with Indian benchmark defaults (Rf=7%, ERP=6.5%).
   Capital structure weighted by Market Cap and Total Debt + Leases.
   Explicit validation error raised if Terminal Growth (g) >= WACC (no silent clamping).
4. Sustainable Terminal Value:
   Terminal FCFF incorporates the fundamental reinvestment rate required to sustain growth:
   Reinvestment_Rate = g / Terminal_ROIC
   Terminal Value contribution is monitored with warnings when dominating (> 75% of EV).
5. Comprehensive Enterprise-to-Equity Bridge:
   Equity Value = EV + Cash + Liquid Investments - Total Debt - Lease Liabilities
                  - Minority Interest - Preferred Equity.
   Rejects missing or invalid share counts rather than inventing denominators.
6. Financial Institutions Routing:
   Refuses industrial FCFF/WACC DCF on Banks/NBFCs; provides Residual Income and Justified P/B models.
7. Scenario Analysis & Reverse DCF:
   Bear/Base/Bull cases and market-implied growth solver.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from indiequity.config import (
    CRORE_DIVISOR,
    DEFAULT_EQUITY_RISK_PREMIUM,
    DEFAULT_RISK_FREE_RATE,
    DEFAULT_TAX_RATE,
    DEFAULT_TERMINAL_GROWTH_RATE,
)
from indiequity.data import FinancialProfile
from indiequity.ratios import CapitalStructure, RatioMetrics, calculate_ratios


class InvalidValuationInputError(ValueError):
    """Raised when valuation inputs violate financial or mathematical validity."""
    pass


class UnsupportedValuationModelError(ValueError):
    """Raised when an industrial DCF model is inappropriately applied to financial institutions."""
    pass


@dataclass
class AnnualForecast:
    """Detailed annual operational, reinvestment, and cash flow forecast."""
    year: int
    revenue: float
    revenue_growth: float
    operating_margin: float
    ebit: float
    operating_taxes: float
    nopat: float
    depreciation_and_amortization: float
    capex: float
    change_in_nwc: float
    fcff: float
    reinvestment: float
    reinvestment_rate: float
    discount_factor: float
    pv_fcff: float


@dataclass
class EnterpriseToEquityBridge:
    """Explicit audit trail bridging Enterprise Value to Equity Value and Per-Share Value."""
    sum_pv_fcff: float
    terminal_value: float
    pv_terminal_value: float
    enterprise_value: float
    terminal_value_pct_of_ev: float
    cash_and_equivalents: float
    short_term_investments: float
    total_liquid_assets: float
    short_term_debt: float
    long_term_debt: float
    total_interest_bearing_debt: float
    lease_liabilities: float
    minority_interest: float
    preferred_stock: float
    implied_equity_value: float
    diluted_shares: float
    intrinsic_value_per_share: float
    current_share_price: float
    margin_of_safety_pct: float


@dataclass
class ForecastAssumptions:
    """User-editable, traceable assumptions for explicit financial projections."""
    forecast_years: int = 5
    revenue_growth_rates: Optional[List[float]] = None
    target_operating_margin: Optional[float] = None
    effective_tax_rate: float = DEFAULT_TAX_RATE
    capex_percent_revenue: Optional[float] = None
    dna_percent_revenue: Optional[float] = None
    nwc_percent_revenue: Optional[float] = None
    terminal_growth_rate: float = DEFAULT_TERMINAL_GROWTH_RATE
    terminal_roic: Optional[float] = None  # If None, assumes Terminal ROIC = WACC
    wacc_override: Optional[float] = None
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE
    equity_risk_premium: float = DEFAULT_EQUITY_RISK_PREMIUM
    beta_override: Optional[float] = None
    cost_of_debt_override: Optional[float] = None


@dataclass
class DCFResult:
    """Complete, fully audited DCF valuation result with forecasts and sensitivities."""
    symbol: str
    company_name: str
    valuation_date: Optional[str]
    model_type: str
    annual_forecasts: List[AnnualForecast]
    bridge: EnterpriseToEquityBridge
    wacc: float
    cost_of_equity: float
    cost_of_debt_after_tax: float
    capital_structure_weight_equity: float
    capital_structure_weight_debt: float
    terminal_growth_rate: float
    terminal_roic: float
    terminal_reinvestment_rate: float
    sensitivity_table: pd.DataFrame
    assumptions: ForecastAssumptions
    warnings: List[str] = field(default_factory=list)

    # Backward-compatible convenience properties
    @property
    def current_price(self) -> float:
        return self.bridge.current_share_price

    @property
    def shares_outstanding(self) -> float:
        return self.bridge.diluted_shares

    @property
    def intrinsic_value_per_share(self) -> float:
        return self.bridge.intrinsic_value_per_share

    @property
    def upside_downside_pct(self) -> float:
        return self.bridge.margin_of_safety_pct

    @property
    def sum_pv_fcff(self) -> float:
        return self.bridge.sum_pv_fcff

    @property
    def terminal_value(self) -> float:
        return self.bridge.terminal_value

    @property
    def pv_terminal_value(self) -> float:
        return self.bridge.pv_terminal_value

    @property
    def enterprise_value(self) -> float:
        return self.bridge.enterprise_value

    @property
    def total_debt(self) -> float:
        return self.bridge.total_interest_bearing_debt + self.bridge.lease_liabilities

    @property
    def cash_and_equivalents(self) -> float:
        return self.bridge.total_liquid_assets

    @property
    def implied_equity_value(self) -> float:
        return self.bridge.implied_equity_value

    @property
    def projected_fcff(self) -> List[float]:
        return [f.fcff for f in self.annual_forecasts]

    @property
    def pv_projected_fcff(self) -> List[float]:
        return [f.pv_fcff for f in self.annual_forecasts]


@dataclass
class ResidualIncomeResult:
    """Valuation result for Financial Institutions (Banks/NBFCs) using Residual Income and Justified P/B."""
    symbol: str
    company_name: str
    model_type: str  # "Residual Income Model (Banks/NBFCs)"
    current_share_price: float
    book_value_per_share: float
    normalized_roe: float
    cost_of_equity: float
    sustainable_growth_rate: float
    justified_pb_ratio: float
    justified_price_per_share: float
    projected_residual_income: List[float]
    intrinsic_value_per_share: float
    margin_of_safety_pct: float
    warnings: List[str] = field(default_factory=list)


def calculate_fcff_item(
    ebit: float,
    tax_rate: float,
    depreciation_and_amortization: float,
    capex: float,
    change_in_operating_nwc: float,
) -> float:
    """
    Compute Free Cash Flow to Firm (FCFF) according to core corporate finance identity:
    FCFF = EBIT * (1 - tax_rate) + D&A - CapEx - Change_in_Operating_NWC

    Worked Example:
    EBIT = 200 Cr, Tax = 25%, D&A = 40 Cr, CapEx = 70 Cr, Change in NWC = 20 Cr:
    FCFF = 200 * (1 - 0.25) + 40 - 70 - 20 = 150 + 40 - 70 - 20 = 100 Cr.
    """
    nopat = ebit * (1.0 - tax_rate)
    return nopat + depreciation_and_amortization - capex - change_in_operating_nwc


def estimate_wacc(
    profile: FinancialProfile,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    equity_risk_premium: float = DEFAULT_EQUITY_RISK_PREMIUM,
    tax_rate: float = DEFAULT_TAX_RATE,
    beta_override: Optional[float] = None,
    cost_of_debt_override: Optional[float] = None,
    wacc_override: Optional[float] = None,
) -> Tuple[float, float, float, float, float]:
    """
    Estimate market-value Weighted Average Cost of Capital (WACC).

    Returns:
        Tuple of (wacc, cost_of_equity, cost_of_debt_after_tax, weight_equity, weight_debt)
    """
    if wacc_override is not None:
        if wacc_override <= 0.0 or wacc_override > 0.50:
            raise InvalidValuationInputError(f"WACC override ({wacc_override:.2%}) is outside plausible economic range (0% - 50%).")
        return wacc_override, wacc_override, 0.0, 1.0, 0.0

    # 1. Cost of Equity (CAPM)
    beta = beta_override if beta_override is not None else (profile.beta if (profile.beta is not None and profile.beta > 0) else 1.05)
    cost_of_equity = risk_free_rate + (beta * equity_risk_premium)

    # 2. Capital Structure Weights (Market Value of Equity + Total Debt & Leases)
    metrics = calculate_ratios(profile)
    market_equity = metrics.capital_structure.market_cap
    if market_equity <= 0:
        market_equity = max(metrics.capital_structure.total_stockholders_equity, 10_000_000.0)

    total_debt_and_leases = metrics.capital_structure.total_debt_and_leases

    # 3. Cost of Debt
    if cost_of_debt_override is not None:
        cost_of_debt = cost_of_debt_override
    else:
        inc = profile.income_stmt
        interest = abs(profile.get_latest(inc, ["Interest Expense", "Finance Costs"], default=0.0))
        if total_debt_and_leases > 0 and interest > 0:
            cost_of_debt = interest / total_debt_and_leases
            # Sanity bound for Indian corporate debt (7.0% - 15.0%)
            cost_of_debt = max(0.070, min(cost_of_debt, 0.150))
        else:
            cost_of_debt = risk_free_rate + 0.020  # Benchmark Indian corporate spread over G-Sec: 200 bps

    cost_of_debt_after_tax = cost_of_debt * (1.0 - tax_rate)

    total_capital = market_equity + total_debt_and_leases
    w_equity = market_equity / total_capital
    w_debt = total_debt_and_leases / total_capital

    wacc = (w_equity * cost_of_equity) + (w_debt * cost_of_debt_after_tax)
    return wacc, cost_of_equity, cost_of_debt_after_tax, w_equity, w_debt


def calculate_dcf(
    profile: FinancialProfile,
    forecast_years: int = 5,
    assumptions: Optional[ForecastAssumptions] = None,
    fcf_growth_rate: Optional[float] = None,  # For backward-compatibility
    wacc: Optional[float] = None,             # For backward-compatibility
    terminal_growth_rate: Optional[float] = None,  # For backward-compatibility
) -> DCFResult:
    """
    Perform rigorous multi-year FCFF DCF valuation with explicit forecasting,
    reinvestment modeling, and sensitivity analysis.
    """
    # Phase 3 Guard: Refuse industrial DCF on Financial Institutions
    if profile.is_financial_institution():
        raise UnsupportedValuationModelError(
            f"Cannot run industrial FCFF/WACC DCF on '{profile.company_name}' ({profile.symbol}). "
            f"It is categorized under '{profile.sector} / {profile.industry}'. "
            "For banks and lending institutions, use the Residual Income Model (RIM) or Justified P/B."
        )

    # Initialize assumptions
    asm = assumptions or ForecastAssumptions(forecast_years=forecast_years)
    if forecast_years != asm.forecast_years:
        asm.forecast_years = forecast_years

    if wacc is not None:
        asm.wacc_override = wacc
    if terminal_growth_rate is not None:
        asm.terminal_growth_rate = terminal_growth_rate

    # Reject invalid inputs
    if asm.forecast_years not in (5, 10):
        raise InvalidValuationInputError(f"Forecast period must be 5 or 10 years, got {asm.forecast_years}.")

    if profile.shares_outstanding <= 0:
        raise InvalidValuationInputError(
            f"Cannot calculate per-share intrinsic value: shares outstanding is missing or non-positive ({profile.shares_outstanding})."
        )

    # Compute baseline metrics
    metrics = calculate_ratios(profile)
    warnings = list(metrics.warnings)

    base_rev = metrics.revenue
    if base_rev <= 0:
        raise InvalidValuationInputError(f"Cannot perform DCF: baseline revenue is missing or non-positive (₹{base_rev}).")

    base_ebit = metrics.ebit
    base_margin = (base_ebit / base_rev) if base_rev > 0 else 0.10
    target_margin = asm.target_operating_margin if asm.target_operating_margin is not None else base_margin

    # Estimate WACC
    calc_wacc, ke, kd_at, w_eq, w_d = estimate_wacc(
        profile,
        risk_free_rate=asm.risk_free_rate,
        equity_risk_premium=asm.equity_risk_premium,
        tax_rate=asm.effective_tax_rate,
        beta_override=asm.beta_override,
        cost_of_debt_override=asm.cost_of_debt_override,
        wacc_override=asm.wacc_override,
    )

    g_term = asm.terminal_growth_rate

    # Rule 3: Explicit validation error if g >= WACC. Do not silently increase WACC!
    if g_term >= calc_wacc:
        raise InvalidValuationInputError(
            f"Invalid valuation inputs: Terminal growth rate ({g_term:.2%}) cannot be greater than or equal to WACC ({calc_wacc:.2%}). "
            "Perpetual Gordon Growth model denominator (WACC - g) would be non-positive."
        )

    # Historical operational ratios for default projections
    capex_pct = asm.capex_percent_revenue if asm.capex_percent_revenue is not None else (
        metrics.capex / base_rev if metrics.capex > 0 else 0.05
    )
    dna_pct = asm.dna_percent_revenue if asm.dna_percent_revenue is not None else (
        metrics.depreciation_and_amortization / base_rev if metrics.depreciation_and_amortization > 0 else 0.03
    )
    nwc_pct = asm.nwc_percent_revenue if asm.nwc_percent_revenue is not None else (
        metrics.working_capital.operating_nwc / base_rev if metrics.working_capital.operating_nwc > 0 else 0.15
    )

    # Setup annual growth schedule
    n_years = asm.forecast_years
    if asm.revenue_growth_rates is not None:
        if len(asm.revenue_growth_rates) < n_years:
            # Pad with last growth rate fading to terminal growth
            last_g = asm.revenue_growth_rates[-1]
            growth_schedule = list(asm.revenue_growth_rates)
            while len(growth_schedule) < n_years:
                last_g = max(g_term, last_g * 0.90)
                growth_schedule.append(last_g)
        else:
            growth_schedule = asm.revenue_growth_rates[:n_years]
    elif fcf_growth_rate is not None:
        growth_schedule = [fcf_growth_rate] * n_years
    else:
        # Default 5-year Indian small/mid-cap growth glide path
        growth_schedule = [0.14, 0.13, 0.12, 0.10, 0.08] if n_years == 5 else [
            0.15, 0.14, 0.13, 0.12, 0.11, 0.10, 0.09, 0.08, 0.07, 0.06
        ]

    # Explicit Annual Forecasts
    forecasts: List[AnnualForecast] = []
    current_rev = base_rev
    current_nwc = metrics.working_capital.operating_nwc
    sum_pv_fcff = 0.0

    for t in range(1, n_years + 1):
        g_t = growth_schedule[t - 1]
        current_rev *= (1.0 + g_t)

        # Margin interpolation towards target margin
        weight_target = t / n_years
        margin_t = (1.0 - weight_target) * base_margin + (weight_target * target_margin)
        ebit_t = current_rev * margin_t
        taxes_t = ebit_t * asm.effective_tax_rate
        nopat_t = ebit_t - taxes_t

        dna_t = current_rev * dna_pct
        capex_t = current_rev * capex_pct

        target_nwc_t = current_rev * nwc_pct
        delta_nwc_t = target_nwc_t - current_nwc
        current_nwc = target_nwc_t

        # Fundamental FCFF: EBIT*(1-t) + D&A - CapEx - ΔNWC
        # Negative FCFF is preserved!
        fcff_t = calculate_fcff_item(ebit_t, asm.effective_tax_rate, dna_t, capex_t, delta_nwc_t)

        reinvestment_t = capex_t - dna_t + delta_nwc_t
        reinvestment_rate_t = (reinvestment_t / nopat_t) if nopat_t != 0 else 0.0

        discount_factor = 1.0 / ((1.0 + calc_wacc) ** t)
        pv_fcff_t = fcff_t * discount_factor
        sum_pv_fcff += pv_fcff_t

        forecasts.append(
            AnnualForecast(
                year=t,
                revenue=current_rev,
                revenue_growth=g_t,
                operating_margin=margin_t,
                ebit=ebit_t,
                operating_taxes=taxes_t,
                nopat=nopat_t,
                depreciation_and_amortization=dna_t,
                capex=capex_t,
                change_in_nwc=delta_nwc_t,
                fcff=fcff_t,
                reinvestment=reinvestment_t,
                reinvestment_rate=reinvestment_rate_t,
                discount_factor=discount_factor,
                pv_fcff=pv_fcff_t,
            )
        )

    # 4. Terminal Value (Stable Growth with Reinvestment)
    # Terminal ROIC: default is competitive equilibrium (Terminal ROIC = WACC)
    term_roic = asm.terminal_roic if asm.terminal_roic is not None else calc_wacc
    if term_roic <= 0:
        term_roic = calc_wacc

    terminal_reinvestment_rate = min(1.0, max(0.0, g_term / term_roic))

    # NOPAT in Year N+1
    nopat_n = forecasts[-1].nopat
    nopat_terminal = nopat_n * (1.0 + g_term)
    terminal_fcff = nopat_terminal * (1.0 - terminal_reinvestment_rate)

    terminal_value = terminal_fcff / (calc_wacc - g_term)
    pv_terminal_value = terminal_value / ((1.0 + calc_wacc) ** n_years)

    # Enterprise Value
    enterprise_value = sum_pv_fcff + pv_terminal_value
    tv_pct_of_ev = (pv_terminal_value / enterprise_value) if enterprise_value > 0 else 0.0

    if tv_pct_of_ev > 0.75:
        warnings.append(
            f"Terminal Value constitutes {tv_pct_of_ev:.1%} of Enterprise Value. "
            "Valuation is highly sensitive to long-term perpetual growth assumptions."
        )

    # 5. Enterprise-to-Equity Value Bridge
    cs = metrics.capital_structure
    cash_liquid = cs.total_cash_and_liquid_assets
    total_debt = cs.total_interest_bearing_debt
    leases = cs.lease_liabilities
    minority_interest = cs.minority_interest
    pref_stock = cs.preferred_stock

    implied_equity_value = enterprise_value + cash_liquid - total_debt - leases - minority_interest - pref_stock
    diluted_shares = profile.shares_outstanding

    intrinsic_value_per_share = implied_equity_value / diluted_shares
    current_price = profile.current_price

    margin_of_safety_pct = (
        ((intrinsic_value_per_share - current_price) / current_price)
        if current_price > 0 else 0.0
    )

    bridge = EnterpriseToEquityBridge(
        sum_pv_fcff=sum_pv_fcff,
        terminal_value=terminal_value,
        pv_terminal_value=pv_terminal_value,
        enterprise_value=enterprise_value,
        terminal_value_pct_of_ev=tv_pct_of_ev,
        cash_and_equivalents=cs.cash_and_equivalents,
        short_term_investments=cs.short_term_investments,
        total_liquid_assets=cash_liquid,
        short_term_debt=cs.short_term_debt,
        long_term_debt=cs.long_term_debt,
        total_interest_bearing_debt=total_debt,
        lease_liabilities=leases,
        minority_interest=minority_interest,
        preferred_stock=pref_stock,
        implied_equity_value=implied_equity_value,
        diluted_shares=diluted_shares,
        intrinsic_value_per_share=intrinsic_value_per_share,
        current_share_price=current_price,
        margin_of_safety_pct=margin_of_safety_pct,
    )

    # 6. Sensitivity Table (WACC vs Terminal Growth Rate)
    sensitivity_table = generate_sensitivity_matrix(
        forecasts=forecasts,
        base_rev=base_rev,
        base_margin=base_margin,
        target_margin=target_margin,
        n_years=n_years,
        growth_schedule=growth_schedule,
        tax_rate=asm.effective_tax_rate,
        capex_pct=capex_pct,
        dna_pct=dna_pct,
        nwc_pct=nwc_pct,
        center_wacc=calc_wacc,
        center_g=g_term,
        term_roic=term_roic,
        cash_liquid=cash_liquid,
        total_debt_and_claims=total_debt + leases + minority_interest + pref_stock,
        shares=diluted_shares,
    )

    return DCFResult(
        symbol=profile.symbol,
        company_name=profile.company_name,
        valuation_date=profile.latest_period_date,
        model_type="5-Year FCFF DCF (Stable Growth Terminal Value)" if n_years == 5 else "10-Year Transition FCFF DCF",
        annual_forecasts=forecasts,
        bridge=bridge,
        wacc=calc_wacc,
        cost_of_equity=ke,
        cost_of_debt_after_tax=kd_at,
        capital_structure_weight_equity=w_eq,
        capital_structure_weight_debt=w_d,
        terminal_growth_rate=g_term,
        terminal_roic=term_roic,
        terminal_reinvestment_rate=terminal_reinvestment_rate,
        sensitivity_table=sensitivity_table,
        assumptions=asm,
        warnings=warnings,
    )


def generate_sensitivity_matrix(
    forecasts: List[AnnualForecast],
    base_rev: float,
    base_margin: float,
    target_margin: float,
    n_years: int,
    growth_schedule: List[float],
    tax_rate: float,
    capex_pct: float,
    dna_pct: float,
    nwc_pct: float,
    center_wacc: float,
    center_g: float,
    term_roic: float,
    cash_liquid: float,
    total_debt_and_claims: float,
    shares: float,
) -> pd.DataFrame:
    """
    Construct a 2D sensitivity grid varying WACC (rows) and Terminal Growth Rate (columns).
    Cells contain the resulting Intrinsic Value per share (₹).
    """
    wacc_steps = [center_wacc - 0.02, center_wacc - 0.01, center_wacc, center_wacc + 0.01, center_wacc + 0.02]
    g_steps = [center_g - 0.015, center_g - 0.0075, center_g, center_g + 0.0075, center_g + 0.015]

    grid: Dict[str, List[float]] = {}
    row_labels = [f"WACC {w * 100:.1f}%" for w in wacc_steps]

    for g in g_steps:
        col_label = f"g = {g * 100:.2f}%"
        col_vals = []
        for w in wacc_steps:
            if w <= g:
                col_vals.append(float("nan"))
                continue

            # Recalculate PV of cash flows for new WACC
            sum_pv = 0.0
            for f in forecasts:
                dfact = 1.0 / ((1.0 + w) ** f.year)
                sum_pv += f.fcff * dfact

            # Terminal Value
            term_reinv = min(1.0, max(0.0, g / term_roic)) if term_roic > 0 else (g / w)
            nopat_terminal = forecasts[-1].nopat * (1.0 + g)
            term_fcff = nopat_terminal * (1.0 - term_reinv)
            tv = term_fcff / (w - g)
            pv_tv = tv / ((1.0 + w) ** n_years)

            ev = sum_pv + pv_tv
            eq_val = ev + cash_liquid - total_debt_and_claims
            per_share = eq_val / shares if shares > 0 else 0.0
            col_vals.append(round(per_share, 2))

        grid[col_label] = col_vals

    return pd.DataFrame(grid, index=row_labels)


def calculate_residual_income_valuation(
    profile: FinancialProfile,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    equity_risk_premium: float = DEFAULT_EQUITY_RISK_PREMIUM,
    forecast_years: int = 5,
    terminal_growth_rate: float = DEFAULT_TERMINAL_GROWTH_RATE,
) -> ResidualIncomeResult:
    """
    Residual Income Model (RIM) and Justified Price-to-Book for Financial Institutions (Banks/NBFCs).

    Formulas:
    1. Cost of Equity Ke = Rf + Beta * ERP
    2. Residual Income_t = (ROE_t - Ke) * Book_Value_{t-1}
    3. Terminal Residual Income = RI_5 * (1 + g) / (Ke - g)
    4. Intrinsic Equity Value = Book_Value_0 + Sum(PV(RI)) + PV(Terminal RI)
    5. Justified P/B = (ROE - g) / (Ke - g)
    """
    bs = profile.balance_sheet
    inc = profile.income_stmt
    warnings = list(profile.data_quality_warnings)

    total_equity = profile.get_latest(bs, [
        "Stockholders Equity", "Total Stockholder Equity", "Common Stock Equity", "Total Equity"
    ], default=0.0)

    if total_equity <= 0:
        raise InvalidValuationInputError(f"Cannot perform bank valuation: Book value of equity is missing or non-positive.")

    shares = profile.shares_outstanding
    if shares <= 0:
        raise InvalidValuationInputError(f"Shares outstanding is missing or non-positive.")

    bvps = total_equity / shares

    # Historical normalized Net Income and ROE
    net_income = profile.get_latest(inc, ["Net Income Common Stockholders", "Net Income"], default=0.0)
    current_roe = (net_income / total_equity) if total_equity > 0 else 0.14
    normalized_roe = max(0.08, min(current_roe, 0.22))  # Sustainable bank ROE

    beta = profile.beta if (profile.beta is not None and profile.beta > 0) else 1.10
    ke = risk_free_rate + (beta * equity_risk_premium)

    g = terminal_growth_rate
    if g >= ke:
        raise InvalidValuationInputError(f"Terminal growth rate ({g:.2%}) cannot be >= Cost of Equity ({ke:.2%}).")

    # 1. Justified P/B
    justified_pb = (normalized_roe - g) / (ke - g) if (ke - g) > 0 else 1.0
    justified_pb = max(0.5, justified_pb)
    justified_price = justified_pb * bvps

    # 2. Multi-year Residual Income Forecast
    retention_ratio = 1.0 - (g / normalized_roe) if normalized_roe > 0 else 0.50
    current_bv = total_equity
    pv_ri_sum = 0.0
    ri_schedule = []

    for t in range(1, forecast_years + 1):
        ri_t = (normalized_roe - ke) * current_bv
        pv_ri_t = ri_t / ((1.0 + ke) ** t)
        pv_ri_sum += pv_ri_t
        ri_schedule.append(ri_t)
        # Book value accretion through retained earnings
        earnings_t = normalized_roe * current_bv
        retained_t = earnings_t * retention_ratio
        current_bv += retained_t

    # Terminal Residual Income
    terminal_ri = (ri_schedule[-1] * (1.0 + g)) / (ke - g)
    pv_terminal_ri = terminal_ri / ((1.0 + ke) ** forecast_years)

    intrinsic_equity = total_equity + pv_ri_sum + pv_terminal_ri
    intrinsic_val_per_share = intrinsic_equity / shares

    current_price = profile.current_price
    margin_of_safety = (
        ((intrinsic_val_per_share - current_price) / current_price)
        if current_price > 0 else 0.0
    )

    return ResidualIncomeResult(
        symbol=profile.symbol,
        company_name=profile.company_name,
        model_type="Residual Income Model (Banks/NBFCs)",
        current_share_price=current_price,
        book_value_per_share=bvps,
        normalized_roe=normalized_roe,
        cost_of_equity=ke,
        sustainable_growth_rate=g,
        justified_pb_ratio=justified_pb,
        justified_price_per_share=justified_price,
        projected_residual_income=ri_schedule,
        intrinsic_value_per_share=intrinsic_val_per_share,
        margin_of_safety_pct=margin_of_safety,
        warnings=warnings,
    )


def reverse_dcf(
    profile: FinancialProfile,
    target_price: Optional[float] = None,
    wacc: Optional[float] = None,
    terminal_growth_rate: float = DEFAULT_TERMINAL_GROWTH_RATE,
) -> Dict[str, Union[float, str]]:
    """
    Reverse DCF: Solves for the implied 5-year revenue growth rate
    required to justify the current market price P_0.
    """
    price_to_match = target_price if target_price is not None else profile.current_price
    if price_to_match <= 0:
        raise InvalidValuationInputError("Target market price must be positive for reverse DCF.")

    # Binary search over 5-year growth rates (-20% to +80%)
    low_g = -0.20
    high_g = 0.80
    best_g = None

    for _ in range(40):
        mid_g = (low_g + high_g) / 2.0
        asm = ForecastAssumptions(
            forecast_years=5,
            revenue_growth_rates=[mid_g] * 5,
            wacc_override=wacc,
            terminal_growth_rate=terminal_growth_rate,
        )
        res = calculate_dcf(profile, assumptions=asm)
        val = res.intrinsic_value_per_share

        if abs(val - price_to_match) < 0.25:
            best_g = mid_g
            break
        elif val < price_to_match:
            low_g = mid_g
        else:
            high_g = mid_g

    implied_g = best_g if best_g is not None else mid_g
    assessment = "Demanding / Priced for Perfection" if implied_g > 0.20 else (
        "Reasonable Growth Embedded" if implied_g > 0.08 else "Low Growth / Value Expectations"
    )

    return {
        "symbol": profile.symbol,
        "market_price": price_to_match,
        "implied_5y_cagr": round(implied_g, 4),
        "implied_5y_cagr_pct": f"{implied_g * 100:.2f}%",
        "market_expectation_assessment": assessment,
    }


def run_scenario_analysis(
    profile: FinancialProfile,
    base_growth: Optional[float] = None,
) -> Dict[str, DCFResult]:
    """
    Run Bear, Base, and Bull valuation scenarios with explicit assumption deviations.
    """
    r = calculate_ratios(profile)
    base_margin = (r.ebit / r.revenue) if r.revenue > 0 else 0.12
    default_g = base_growth if base_growth is not None else 0.12

    # 1. Bear Case (Lower growth, margin compression, higher cost of capital)
    bear_asm = ForecastAssumptions(
        forecast_years=5,
        revenue_growth_rates=[max(0.04, default_g - 0.05)] * 5,
        target_operating_margin=max(0.06, base_margin - 0.025),
        terminal_growth_rate=0.04,
        wacc_override=estimate_wacc(profile)[0] + 0.015,
    )
    bear_res = calculate_dcf(profile, assumptions=bear_asm)

    # 2. Base Case
    base_asm = ForecastAssumptions(
        forecast_years=5,
        revenue_growth_rates=[default_g] * 5,
        target_operating_margin=base_margin,
        terminal_growth_rate=0.05,
    )
    base_res = calculate_dcf(profile, assumptions=base_asm)

    # 3. Bull Case (Acceleration, margin expansion, lower risk premium)
    bull_asm = ForecastAssumptions(
        forecast_years=5,
        revenue_growth_rates=[default_g + 0.05] * 5,
        target_operating_margin=base_margin + 0.025,
        terminal_growth_rate=0.055,
        wacc_override=max(0.09, estimate_wacc(profile)[0] - 0.01),
    )
    bull_res = calculate_dcf(profile, assumptions=bull_asm)

    return {
        "Bear": bear_res,
        "Base": base_res,
        "Bull": bull_res,
    }
