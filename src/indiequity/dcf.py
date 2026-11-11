"""
Discounted Cash Flow (DCF) valuation engine with 2D sensitivity analysis.

Implements a 5-Year Free Cash Flow to Firm (FCFF) model:
- WACC calculation using CAPM with Indian benchmark defaults
- 5-Year explicit forecast period
- Gordon Growth Model for Terminal Value
- Enterprise Value to Implied Equity Value bridge
- 2D Sensitivity Matrix: Discount Rate (WACC) vs Terminal Growth Rate (g)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
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
from indiequity.ratios import calculate_ratios


@dataclass
class DCFResult:
    """Complete summary of DCF valuation and intrinsic value per share."""
    symbol: str
    company_name: str
    current_price: float
    shares_outstanding: float
    base_fcf: float
    fcf_growth_rate: float
    projected_fcff: List[float]
    pv_projected_fcff: List[float]
    sum_pv_fcff: float
    wacc: float
    terminal_growth_rate: float
    terminal_value: float
    pv_terminal_value: float
    enterprise_value: float
    total_debt: float
    cash_and_equivalents: float
    implied_equity_value: float
    intrinsic_value_per_share: float
    upside_downside_pct: float
    sensitivity_table: pd.DataFrame


def estimate_wacc(
    profile: FinancialProfile,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    equity_risk_premium: float = DEFAULT_EQUITY_RISK_PREMIUM,
    tax_rate: float = DEFAULT_TAX_RATE,
    cost_of_debt: Optional[float] = None,
) -> float:
    """
    Estimate Weighted Average Cost of Capital (WACC) using the Capital Asset Pricing Model (CAPM).

    Formula:
        Ke = Rf + Beta * ERP
        Kd_after_tax = Kd * (1 - t)
        WACC = (E / (D + E)) * Ke + (D / (D + E)) * Kd_after_tax

    Args:
        profile: FinancialProfile containing statements and market data.
        risk_free_rate: Indian 10Y G-Sec yield (default: 7.0%).
        equity_risk_premium: Long-term Indian ERP (default: 6.5%).
        tax_rate: Corporate tax rate (default: 25.17%).
        cost_of_debt: Pre-tax borrowing rate. If None, estimated from financial statements.

    Returns:
        Estimated WACC as a decimal (e.g. 0.115 for 11.5%).
    """
    beta = profile.beta if (profile.beta is not None and profile.beta > 0) else 1.05
    cost_of_equity = risk_free_rate + (beta * equity_risk_premium)

    metrics = calculate_ratios(profile)
    total_equity = metrics.market_cap if metrics.market_cap > 0 else (metrics.total_equity if metrics.total_equity > 0 else 1.0)
    total_debt = metrics.total_debt

    # Estimate cost of debt if not provided
    if cost_of_debt is None:
        inc = profile.income_stmt
        interest = abs(profile.get_latest(inc, ["Interest Expense", "Finance Costs"]))
        if total_debt > 0 and interest > 0:
            cost_of_debt = interest / total_debt
            # Cap within realistic Indian borrowing rates (7.5% - 14%)
            cost_of_debt = max(0.075, min(cost_of_debt, 0.14))
        else:
            cost_of_debt = 0.09  # 9.0% default Indian corporate borrowing rate

    total_capital = total_equity + total_debt
    w_equity = total_equity / total_capital
    w_debt = total_debt / total_capital

    wacc = (w_equity * cost_of_equity) + (w_debt * cost_of_debt * (1 - tax_rate))
    # Safety boundary: Indian small-cap WACC typically falls between 9% and 18%
    return float(max(0.08, min(wacc, 0.20)))


def calculate_dcf(
    profile: FinancialProfile,
    forecast_years: int = 5,
    fcf_growth_rate: Optional[float] = None,
    wacc: Optional[float] = None,
    terminal_growth_rate: float = DEFAULT_TERMINAL_GROWTH_RATE,
) -> DCFResult:
    """
    Perform 5-Year FCFF DCF Valuation with a 2D sensitivity grid.

    Formulas:
    1. FCFF_t = Base_FCFF * (1 + g)^t
    2. PV(FCFF_t) = FCFF_t / (1 + WACC)^t
    3. Terminal Value = FCFF_5 * (1 + g_term) / (WACC - g_term)
    4. PV(Terminal Value) = Terminal Value / (1 + WACC)^5
    5. Enterprise Value = Sum(PV_FCFF) + PV(Terminal Value)
    6. Implied Equity Value = Enterprise Value + Cash - Debt
    7. Intrinsic Value Per Share = Implied Equity Value / Shares Outstanding

    Args:
        profile: Populated FinancialProfile.
        forecast_years: Forecast period (default: 5 years).
        fcf_growth_rate: Expected annual FCF growth for 5 years. If None, uses conservative 10%.
        wacc: Discount rate. If None, estimated via CAPM.
        terminal_growth_rate: Long-term perpetual GDP growth (default: 5.0%).

    Returns:
        DCFResult object.
    """
    metrics = calculate_ratios(profile)

    # Base year Free Cash Flow (fallback to 12% of revenue if negative/zero)
    base_fcf = metrics.free_cash_flow
    if base_fcf <= 0:
        base_fcf = max(metrics.revenue * 0.10, 10_000_000.0)

    # Growth rate
    if fcf_growth_rate is None:
        fcf_growth_rate = 0.12  # Conservative 12% Indian small-cap compound rate

    # WACC
    if wacc is None:
        discount_rate = estimate_wacc(profile)
    else:
        discount_rate = wacc

    # Ensure WACC > terminal growth rate for Gordon Growth model validity
    if discount_rate <= terminal_growth_rate:
        discount_rate = terminal_growth_rate + 0.03

    # Project FCFF for 5 years
    projected_fcff = []
    pv_projected_fcff = []
    current_fcf = base_fcf

    for year in range(1, forecast_years + 1):
        current_fcf *= (1 + fcf_growth_rate)
        pv = current_fcf / ((1 + discount_rate) ** year)
        projected_fcff.append(current_fcf)
        pv_projected_fcff.append(pv)

    sum_pv_fcff = sum(pv_projected_fcff)

    # Terminal Value (Gordon Growth)
    terminal_fcf = projected_fcff[-1] * (1 + terminal_growth_rate)
    terminal_value = terminal_fcf / (discount_rate - terminal_growth_rate)
    pv_terminal_value = terminal_value / ((1 + discount_rate) ** forecast_years)

    # Enterprise Value & Equity Value Bridge
    enterprise_value = sum_pv_fcff + pv_terminal_value
    total_debt = metrics.total_debt
    cash = metrics.cash_and_equivalents
    implied_equity_value = enterprise_value + cash - total_debt

    # Per Share Intrinsic Value
    shares = profile.shares_outstanding
    if shares <= 0:
        # Fallback: estimate shares from market cap and price
        shares = (metrics.market_cap / profile.current_price) if profile.current_price > 0 else 1.0

    intrinsic_value_per_share = implied_equity_value / shares
    current_price = profile.current_price

    upside_downside_pct = (
        ((intrinsic_value_per_share - current_price) / current_price)
        if current_price > 0 else 0.0
    )

    # Generate 2D Sensitivity Analysis Table (WACC vs Terminal Growth Rate)
    sensitivity_table = generate_sensitivity_matrix(
        base_fcf=base_fcf,
        forecast_years=forecast_years,
        fcf_growth_rate=fcf_growth_rate,
        center_wacc=discount_rate,
        center_terminal_growth=terminal_growth_rate,
        total_debt=total_debt,
        cash=cash,
        shares_outstanding=shares,
    )

    return DCFResult(
        symbol=profile.symbol,
        company_name=profile.company_name,
        current_price=current_price,
        shares_outstanding=shares,
        base_fcf=base_fcf,
        fcf_growth_rate=fcf_growth_rate,
        projected_fcff=projected_fcff,
        pv_projected_fcff=pv_projected_fcff,
        sum_pv_fcff=sum_pv_fcff,
        wacc=discount_rate,
        terminal_growth_rate=terminal_growth_rate,
        terminal_value=terminal_value,
        pv_terminal_value=pv_terminal_value,
        enterprise_value=enterprise_value,
        total_debt=total_debt,
        cash_and_equivalents=cash,
        implied_equity_value=implied_equity_value,
        intrinsic_value_per_share=intrinsic_value_per_share,
        upside_downside_pct=upside_downside_pct,
        sensitivity_table=sensitivity_table,
    )


def generate_sensitivity_matrix(
    base_fcf: float,
    forecast_years: int,
    fcf_growth_rate: float,
    center_wacc: float,
    center_terminal_growth: float,
    total_debt: float,
    cash: float,
    shares_outstanding: float,
) -> pd.DataFrame:
    """
    Construct a 2D sensitivity grid varying WACC (rows) and Terminal Growth Rate (columns).
    Cells represent the resulting Intrinsic Value Per Share (₹).
    """
    # WACC steps: center +/- 2% (5 rows)
    wacc_steps = [center_wacc - 0.02, center_wacc - 0.01, center_wacc, center_wacc + 0.01, center_wacc + 0.02]
    # Terminal growth steps: center +/- 1.5% (5 columns)
    g_steps = [
        center_terminal_growth - 0.015,
        center_terminal_growth - 0.0075,
        center_terminal_growth,
        center_terminal_growth + 0.0075,
        center_terminal_growth + 0.015,
    ]

    grid: Dict[str, List[float]] = {}
    row_labels = [f"WACC {w * 100:.1f}%" for w in wacc_steps]

    for g in g_steps:
        col_label = f"g = {g * 100:.2f}%"
        values_in_col = []
        for w in wacc_steps:
            if w <= g:
                values_in_col.append(float("nan"))
                continue

            # 5-Year PV sum
            pv_sum = 0.0
            cf = base_fcf
            for y in range(1, forecast_years + 1):
                cf *= (1 + fcf_growth_rate)
                pv_sum += cf / ((1 + w) ** y)

            # Terminal Value
            term_val = (cf * (1 + g)) / (w - g)
            pv_term = term_val / ((1 + w) ** forecast_years)
            ev = pv_sum + pv_term
            eq_val = ev + cash - total_debt
            val_per_share = eq_val / shares_outstanding
            values_in_col.append(round(val_per_share, 2))

        grid[col_label] = values_in_col

    return pd.DataFrame(grid, index=row_labels)
