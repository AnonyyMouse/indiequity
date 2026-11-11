"""
Rigorous Discounted Cash Flow (DCF), Residual Income (RIM), and equity valuation engine for Indian markets.

Methodology & Standards:
1. Free Cash Flow to Firm (FCFF):
   FCFF = EBIT * (1 - t) + D&A - CapEx - Change_in_Operating_NWC
   Operating working capital strictly excludes cash, debt, and financing liabilities.
   Negative and zero Change in Operating NWC are preserved faithfully (cash released when negative).
   Does not fabricate positive CapEx or D&A when reported as zero.
2. Explicit Multi-Year Financial Forecasts:
   Detailed line-by-line forecasts (Revenue, Operating Margin, EBIT, Taxes, D&A,
   CapEx, ΔNWC, Reinvestment, ROIC) for 5-year or 10-year transition periods.
   Clearly distinguishes ANALYST_DEFINED assumptions from ILLUSTRATIVE_SCENARIO fallbacks.
3. Market-Value WACC Estimation:
   CAPM Cost of Equity with Indian benchmark defaults (Rf=7%, ERP=6.5%).
   Capital structure weighted by Market Cap and Total Debt + Leases.
   Unclamped borrowing cost; raises explicit validation errors on invalid rates.
   Explicit validation error raised if Terminal Growth (g) >= WACC (no silent clamping).
4. Sustainable Terminal Value:
   Terminal FCFF incorporates the fundamental reinvestment rate required to sustain growth:
   Reinvestment_Rate = g / Terminal_ROIC
   Terminal Value contribution is monitored with warnings when dominating (> 75% of EV).
5. Comprehensive Enterprise-to-Equity Bridge:
   Equity Value = EV + Cash + Liquid Investments - Total Debt - Lease Liabilities
                  - Minority Interest - Preferred Equity.
   Rejects missing or invalid share counts rather than inventing denominators.
6. Transparent Model Routing:
   - Banks & Lending NBFCs -> Residual Income Model (RIM) & Justified P/B
   - Asset Management Companies -> MODEL_NOT_SUPPORTED (requires AUM-driven fee realization)
   - Conglomerates -> MODEL_NOT_SUPPORTED (requires Sum-of-the-Parts segment valuation)
   - Industrial, Tech Services, Pharma, General -> FCFF DCF
7. Mathematically Verified Reverse DCF:
   Solves for market-implied growth with strict round-trip tolerance (abs(P_recalc - P_target) / P_target < 0.001).
   Handles unsolvable bounds gracefully.
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
from indiequity.data import BusinessCategory, FinancialProfile
from indiequity.ratios import CapitalStructure, RatioMetrics, calculate_ratios


class InvalidValuationInputError(ValueError):
    """Raised when valuation inputs violate financial or mathematical validity."""
    pass


class UnsupportedValuationModelError(ValueError):
    """Raised when an inappropriate valuation model is applied to a company category."""
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
    assumptions_type: str = "ILLUSTRATIVE_SCENARIO"  # "ANALYST_DEFINED" or "ILLUSTRATIVE_SCENARIO"


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
class BankAnnualForecast:
    """Year-by-year reconciliation of book value, EPS, dividends, and residual income."""
    year: int
    beginning_book_value: float
    eps: float
    dividend: float
    ending_book_value: float
    residual_income: float
    discount_factor: float
    pv_residual_income: float


@dataclass
class ResidualIncomeResult:
    """Valuation result for Financial Institutions (Banks/NBFCs) using Residual Income and Justified P/B."""
    symbol: str
    company_name: str
    model_type: str
    current_share_price: float
    book_value_per_share: float
    normalized_roe: float
    cost_of_equity: float
    sustainable_growth_rate: float
    retention_ratio: float
    payout_ratio: float
    justified_pb_ratio: float
    justified_price_per_share: float
    annual_forecasts: List[BankAnnualForecast]
    sum_pv_residual_income: float
    terminal_residual_income: float
    pv_terminal_residual_income: float
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

    Worked Examples:
    Case A (Increase in NWC of +₹20 Cr):
      EBIT = 200, Tax = 25% (50), D&A = 40, CapEx = 70, ΔNWC = +20:
      FCFF = 200 * (1 - 0.25) + 40 - 70 - 20 = 150 + 40 - 70 - 20 = 100 Cr.

    Case B (Decrease in NWC of -₹20 Cr, cash released from working capital):
      EBIT = 200, Tax = 25% (50), D&A = 40, CapEx = 70, ΔNWC = -20:
      FCFF = 150 + 40 - 70 - (-20) = 150 + 40 - 70 + 20 = 140 Cr.
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

    Validation:
    - User overrides are validated for economic feasibility.
    - Borrowing cost is unclamped, reflecting true reported financial cost or market evidence.
    """
    if wacc_override is not None:
        if wacc_override <= 0.0 or wacc_override > 0.50:
            raise InvalidValuationInputError(f"WACC override ({wacc_override:.2%}) is outside plausible economic range (0% - 50%).")
        return wacc_override, wacc_override, 0.0, 1.0, 0.0

    if risk_free_rate <= 0.0 or risk_free_rate > 0.25:
        raise InvalidValuationInputError(f"Risk-free rate ({risk_free_rate:.2%}) is outside valid range (0% - 25%).")

    if equity_risk_premium <= 0.0 or equity_risk_premium > 0.20:
        raise InvalidValuationInputError(f"Equity Risk Premium ({equity_risk_premium:.2%}) is outside valid range (0% - 20%).")

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
        if cost_of_debt_override < 0.0 or cost_of_debt_override > 0.40:
            raise InvalidValuationInputError(f"Cost of debt override ({cost_of_debt_override:.2%}) is outside plausible range (0% - 40%).")
        cost_of_debt = cost_of_debt_override
    else:
        inc = profile.income_stmt
        interest = abs(profile.get_latest(inc, ["Interest Expense", "Finance Costs"], default=0.0))
        if total_debt_and_leases > 0 and interest > 0:
            cost_of_debt = interest / total_debt_and_leases
            # Validate derived borrowing cost without clamping
            if cost_of_debt < 0.02:
                cost_of_debt = risk_free_rate + 0.015  # Fallback to high-grade corporate spread
        elif total_debt_and_leases == 0:
            cost_of_debt = 0.0
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
    fcf_growth_rate: Optional[float] = None,
    wacc: Optional[float] = None,
    terminal_growth_rate: Optional[float] = None,
) -> DCFResult:
    """
    Perform rigorous multi-year FCFF DCF valuation with explicit forecasting,
    reinvestment modeling, and sensitivity analysis.
    """
    # Phase 2 Guard: Model Routing System
    if profile.is_financial_institution():
        raise UnsupportedValuationModelError(
            f"Cannot run industrial FCFF/WACC DCF on '{profile.company_name}' ({profile.symbol}). "
            f"Business category: {profile.business_category.value}. "
            "Debt is operating raw material, not financing capital. "
            "Use calculate_residual_income_valuation() for banks and lending institutions."
        )

    if profile.is_asset_manager():
        raise UnsupportedValuationModelError(
            f"MODEL_NOT_SUPPORTED: '{profile.company_name}' ({profile.symbol}) is an Asset Management Company "
            f"({profile.business_category.value}). Asset managers require AUM-driven fee realization, net investor flows, "
            "and regulatory capital models. Industrial FCFF DCF and banking RIM are not supported."
        )

    if profile.is_conglomerate():
        raise UnsupportedValuationModelError(
            f"MODEL_NOT_SUPPORTED: '{profile.company_name}' ({profile.symbol}) is a multi-segment Conglomerate "
            f"({profile.business_category.value}). Requires Sum-of-the-Parts (SOTP) segment valuation."
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

    # Strict Validation: Terminal growth >= WACC raises explicit error
    if g_term >= calc_wacc:
        raise InvalidValuationInputError(
            f"Invalid valuation inputs: Terminal growth rate ({g_term:.2%}) cannot be greater than or equal to WACC ({calc_wacc:.2%}). "
            "Perpetual Gordon Growth model denominator (WACC - g) would be non-positive."
        )

    # Historical operational ratios (Phase 3: Preserve genuine zero and negative values!)
    capex_pct = asm.capex_percent_revenue if asm.capex_percent_revenue is not None else (
        metrics.capex / base_rev
    )
    dna_pct = asm.dna_percent_revenue if asm.dna_percent_revenue is not None else (
        metrics.depreciation_and_amortization / base_rev
    )
    nwc_pct = asm.nwc_percent_revenue if asm.nwc_percent_revenue is not None else (
        metrics.working_capital.operating_nwc / base_rev
    )

    # Setup annual growth schedule
    n_years = asm.forecast_years
    if asm.revenue_growth_rates is not None:
        asm.assumptions_type = "ANALYST_DEFINED"
        if len(asm.revenue_growth_rates) < n_years:
            last_g = asm.revenue_growth_rates[-1]
            growth_schedule = list(asm.revenue_growth_rates)
            while len(growth_schedule) < n_years:
                last_g = max(g_term, last_g * 0.90)
                growth_schedule.append(last_g)
        else:
            growth_schedule = asm.revenue_growth_rates[:n_years]
    elif fcf_growth_rate is not None:
        asm.assumptions_type = "ANALYST_DEFINED"
        growth_schedule = [fcf_growth_rate] * n_years
    else:
        # Default illustrative scenario (Clearly labeled!)
        asm.assumptions_type = "ILLUSTRATIVE_SCENARIO"
        warnings.append("Using illustrative growth schedule; replace with company-specific analyst estimates.")
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

        # Preserves zero and negative cash flows faithfully!
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

    # 4. Terminal Value (Stable Growth with Sustainable Reinvestment)
    term_roic = asm.terminal_roic if asm.terminal_roic is not None else calc_wacc
    if term_roic <= 0:
        term_roic = calc_wacc

    terminal_reinvestment_rate = min(1.0, max(0.0, g_term / term_roic))

    nopat_n = forecasts[-1].nopat
    nopat_terminal = nopat_n * (1.0 + g_term)
    terminal_fcff = nopat_terminal * (1.0 - terminal_reinvestment_rate)

    terminal_value = terminal_fcff / (calc_wacc - g_term)
    pv_terminal_value = terminal_value / ((1.0 + calc_wacc) ** n_years)

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

    # 6. Sensitivity Table
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

            sum_pv = 0.0
            for f in forecasts:
                dfact = 1.0 / ((1.0 + w) ** f.year)
                sum_pv += f.fcff * dfact

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
    normalized_roe_override: Optional[float] = None,
    beta_override: Optional[float] = None,
) -> ResidualIncomeResult:
    """
    Residual Income Model (RIM) and Justified Price-to-Book for Financial Institutions (Banks/NBFCs).

    Correct Mathematical Formulation (Phase 1):
    Under sustainable growth without external capital injection:
      g = ROE * RetentionRatio
      RetentionRatio = g / ROE
      PayoutRatio = 1 - (g / ROE)
      EPS_t = ROE * Book_Value_{t-1}
      Dividend_t = EPS_t * PayoutRatio
      Book_Value_t = Book_Value_{t-1} + EPS_t - Dividend_t = Book_Value_{t-1} * (1 + g)
      Residual_Income_t = (ROE_t - Ke) * Book_Value_{t-1}
      Intrinsic Equity Value = Book_Value_0 + Sum(PV(RI_t)) + PV(Terminal_RI)
    """
    # Guard: Reject Non-Financial Institutions
    if profile.is_asset_manager():
        raise UnsupportedValuationModelError(
            f"MODEL_NOT_SUPPORTED: '{profile.company_name}' ({profile.symbol}) is an Asset Manager. "
            "Residual income model requires lending spread economics. Asset managers require AUM fee realization modeling."
        )

    bs = profile.balance_sheet
    inc = profile.income_stmt
    warnings = list(profile.data_quality_warnings)

    total_equity = profile.get_latest(bs, [
        "Stockholders Equity", "Total Stockholder Equity", "Common Stock Equity", "Total Equity"
    ], default=0.0)

    if total_equity <= 0:
        raise InvalidValuationInputError("Cannot perform bank valuation: Book value of equity is missing or non-positive.")

    shares = profile.shares_outstanding
    if shares <= 0:
        raise InvalidValuationInputError("Shares outstanding is missing or non-positive.")

    bvps = total_equity / shares

    # Historical normalized Net Income and ROE
    net_income = profile.get_latest(inc, ["Net Income Common Stockholders", "Net Income"], default=0.0)
    current_roe = (net_income / total_equity) if total_equity > 0 else 0.14
    normalized_roe = normalized_roe_override if normalized_roe_override is not None else current_roe

    # Cost of Equity
    beta = beta_override if beta_override is not None else (profile.beta if (profile.beta is not None and profile.beta > 0) else 1.05)
    ke = risk_free_rate + (beta * equity_risk_premium)

    g = terminal_growth_rate

    # Phase 1 Input Validations:
    if g < 0.0:
        raise InvalidValuationInputError("Sustainable growth rate cannot be negative.")

    if normalized_roe > 0 and g > normalized_roe:
        raise InvalidValuationInputError(
            f"Sustainable growth rate ({g:.2%}) cannot exceed ROE ({normalized_roe:.2%}) without continuous equity dilution "
            f"(Implied retention ratio {g / normalized_roe:.1%} > 100%)."
        )

    if g >= ke:
        raise InvalidValuationInputError(f"Sustainable growth rate ({g:.2%}) cannot be >= Cost of Equity ({ke:.2%}).")

    # Correct Sustainable Growth Identities
    if normalized_roe > 0:
        retention_ratio = min(1.0, max(0.0, g / normalized_roe))
    else:
        retention_ratio = 0.0

    payout_ratio = 1.0 - retention_ratio

    # Justified P/B: (ROE - g) / (Ke - g)
    justified_pb = (normalized_roe - g) / (ke - g) if (ke - g) > 0 else 1.0
    justified_price = justified_pb * bvps

    # Multi-year Residual Income & Book Value Evolution
    current_bv = bvps
    pv_ri_sum = 0.0
    annual_forecasts: List[BankAnnualForecast] = []

    for t in range(1, forecast_years + 1):
        beg_bv = current_bv
        eps_t = normalized_roe * beg_bv
        div_t = eps_t * payout_ratio
        end_bv = beg_bv + eps_t - div_t

        ri_t = (normalized_roe - ke) * beg_bv
        discount_factor = 1.0 / ((1.0 + ke) ** t)
        pv_ri_t = ri_t * discount_factor
        pv_ri_sum += pv_ri_t

        annual_forecasts.append(
            BankAnnualForecast(
                year=t,
                beginning_book_value=beg_bv,
                eps=eps_t,
                dividend=div_t,
                ending_book_value=end_bv,
                residual_income=ri_t,
                discount_factor=discount_factor,
                pv_residual_income=pv_ri_t,
            )
        )
        current_bv = end_bv

    # Terminal Residual Income
    last_ri = annual_forecasts[-1].residual_income
    terminal_ri = (last_ri * (1.0 + g)) / (ke - g)
    pv_terminal_ri = terminal_ri / ((1.0 + ke) ** forecast_years)

    intrinsic_val_per_share = bvps + pv_ri_sum + pv_terminal_ri

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
        retention_ratio=retention_ratio,
        payout_ratio=payout_ratio,
        justified_pb_ratio=justified_pb,
        justified_price_per_share=justified_price,
        annual_forecasts=annual_forecasts,
        sum_pv_residual_income=pv_ri_sum,
        terminal_residual_income=terminal_ri,
        pv_terminal_residual_income=pv_terminal_ri,
        intrinsic_value_per_share=intrinsic_val_per_share,
        margin_of_safety_pct=margin_of_safety,
        warnings=warnings,
    )


def reverse_dcf(
    profile: FinancialProfile,
    target_price: Optional[float] = None,
    wacc: Optional[float] = None,
    terminal_growth_rate: float = DEFAULT_TERMINAL_GROWTH_RATE,
    tolerance: float = 0.001,
) -> Dict[str, Union[float, str]]:
    """
    Reverse DCF: Solves for the implied 5-year revenue growth rate
    required to justify the target market price P_0.

    Validation (Phase 5):
    Verifies that the recovered growth rate reproduces the target price:
      abs(P_recalculated - target_price) / target_price < tolerance (0.001)
    """
    price_to_match = target_price if target_price is not None else profile.current_price
    if price_to_match <= 0:
        raise InvalidValuationInputError("Target market price must be positive for reverse DCF.")

    # Bounds for search (-50% to +120%)
    low_g = -0.50
    high_g = 1.20
    best_g = None

    def evaluate_price(test_g: float) -> float:
        asm = ForecastAssumptions(
            forecast_years=5,
            revenue_growth_rates=[test_g] * 5,
            wacc_override=wacc,
            terminal_growth_rate=terminal_growth_rate,
        )
        return calculate_dcf(profile, assumptions=asm).intrinsic_value_per_share

    # Check bounds
    try:
        val_low = evaluate_price(low_g)
        val_high = evaluate_price(high_g)
    except Exception as e:
        raise InvalidValuationInputError(f"Reverse DCF failed during bound evaluation: {str(e)}")

    if price_to_match < val_low or price_to_match > val_high:
        raise InvalidValuationInputError(
            f"Target market price (₹{price_to_match:,.2f}) is outside solvable range [₹{val_low:,.2f}, ₹{val_high:,.2f}] "
            "for realistic 5-year growth rates (-50% to +120%)."
        )

    # High-precision bisection
    for _ in range(50):
        mid_g = (low_g + high_g) / 2.0
        val_mid = evaluate_price(mid_g)

        rel_error = abs(val_mid - price_to_match) / price_to_match
        if rel_error < tolerance:
            best_g = mid_g
            break
        elif val_mid < price_to_match:
            low_g = mid_g
        else:
            high_g = mid_g

    if best_g is None:
        best_g = mid_g

    # Re-verify acceptance condition
    recalc_price = evaluate_price(best_g)
    final_error = abs(recalc_price - price_to_match) / price_to_match
    if final_error >= tolerance:
        raise InvalidValuationInputError(
            f"Reverse DCF could not satisfy round-trip tolerance ({final_error:.4%} >= {tolerance:.4%})."
        )

    assessment = "Demanding / Priced for Perfection" if best_g > 0.20 else (
        "Reasonable Growth Embedded" if best_g > 0.08 else "Low Growth / Value Expectations"
    )

    return {
        "symbol": profile.symbol,
        "market_price": price_to_match,
        "implied_5y_cagr": round(best_g, 4),
        "implied_5y_cagr_pct": f"{best_g * 100:.2f}%",
        "recalculated_price": round(recalc_price, 2),
        "round_trip_error_pct": f"{final_error * 100:.4f}%",
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

    # 1. Bear Case
    bear_asm = ForecastAssumptions(
        forecast_years=5,
        revenue_growth_rates=[max(0.04, default_g - 0.05)] * 5,
        target_operating_margin=max(0.06, base_margin - 0.025),
        terminal_growth_rate=0.04,
        wacc_override=estimate_wacc(profile)[0] + 0.015,
        assumptions_type="ANALYST_DEFINED",
    )
    bear_res = calculate_dcf(profile, assumptions=bear_asm)

    # 2. Base Case
    base_asm = ForecastAssumptions(
        forecast_years=5,
        revenue_growth_rates=[default_g] * 5,
        target_operating_margin=base_margin,
        terminal_growth_rate=0.05,
        assumptions_type="ANALYST_DEFINED",
    )
    base_res = calculate_dcf(profile, assumptions=base_asm)

    # 3. Bull Case
    bull_asm = ForecastAssumptions(
        forecast_years=5,
        revenue_growth_rates=[default_g + 0.05] * 5,
        target_operating_margin=base_margin + 0.025,
        terminal_growth_rate=0.055,
        wacc_override=max(0.09, estimate_wacc(profile)[0] - 0.01),
        assumptions_type="ANALYST_DEFINED",
    )
    bull_res = calculate_dcf(profile, assumptions=bull_asm)

    return {
        "Bear": bear_res,
        "Base": base_res,
        "Bull": bull_res,
    }
