"""
DuPont Analysis Engine (3-Stage and 5-Stage decomposition).

Decomposes Return on Equity (ROE) into operational efficiency,
asset utilization, tax efficiency, and capital structure leverage.
"""

from dataclasses import dataclass
from typing import Optional
from indiequity.data import FinancialProfile


@dataclass
class DuPont3Stage:
    """
    3-Stage DuPont Decomposition:
    ROE = Net Profit Margin x Asset Turnover x Equity Multiplier
    """
    symbol: str
    net_profit_margin: float
    asset_turnover: float
    equity_multiplier: float
    calculated_roe: float
    actual_roe: float
    discrepancy: float
    interpretation: str


@dataclass
class DuPont5Stage:
    """
    5-Stage DuPont Decomposition:
    ROE = Tax Burden x Interest Burden x Operating Margin x Asset Turnover x Equity Multiplier
    """
    symbol: str
    tax_burden: float
    interest_burden: float
    operating_margin: float
    asset_turnover: float
    equity_multiplier: float
    calculated_roe: float
    actual_roe: float
    discrepancy: float
    interpretation: str


def decompose_dupont_3stage(profile: FinancialProfile) -> DuPont3Stage:
    """
    Perform 3-Stage DuPont ROE decomposition.

    Formulas:
    1. Net Profit Margin = Net Income / Revenue (Profitability)
    2. Asset Turnover = Revenue / Total Assets (Efficiency)
    3. Equity Multiplier = Total Assets / Total Equity (Leverage)
    ROE = Net Profit Margin * Asset Turnover * Equity Multiplier

    Args:
        profile: FinancialProfile object.

    Returns:
        DuPont3Stage dataclass.
    """
    bs = profile.balance_sheet
    inc = profile.income_stmt

    revenue = profile.get_latest(inc, ["Total Revenue", "Operating Revenue", "Revenue"])
    net_income = profile.get_latest(inc, ["Net Income Common Stockholders", "Net Income"])
    total_assets = profile.get_latest(bs, ["Total Assets"])
    total_equity = profile.get_latest(bs, ["Stockholders Equity", "Total Stockholder Equity", "Common Stock Equity", "Total Equity"])

    npm = (net_income / revenue) if revenue > 0 else 0.0
    at = (revenue / total_assets) if total_assets > 0 else 0.0
    em = (total_assets / total_equity) if total_equity > 0 else 0.0

    calc_roe = npm * at * em
    actual_roe = (net_income / total_equity) if total_equity > 0 else 0.0
    discrepancy = abs(calc_roe - actual_roe)

    # Narrative diagnosis
    drivers = []
    if npm > 0.15:
        drivers.append("Strong pricing power (Net Margin > 15%)")
    elif npm < 0.05:
        drivers.append("Low profit margin (Net Margin < 5%)")

    if at > 1.2:
        drivers.append("High capital velocity (Asset Turnover > 1.2x)")
    elif at < 0.5:
        drivers.append("Capital-intensive asset base (Asset Turnover < 0.5x)")

    if em > 2.5:
        drivers.append("Substantial financial leverage (Equity Multiplier > 2.5x)")
    else:
        drivers.append("Conservative balance sheet leverage (Equity Multiplier <= 2.5x)")

    interpretation = "; ".join(drivers)

    return DuPont3Stage(
        symbol=profile.symbol,
        net_profit_margin=npm,
        asset_turnover=at,
        equity_multiplier=em,
        calculated_roe=calc_roe,
        actual_roe=actual_roe,
        discrepancy=discrepancy,
        interpretation=interpretation,
    )


def decompose_dupont_5stage(profile: FinancialProfile) -> DuPont5Stage:
    """
    Perform 5-Stage DuPont ROE decomposition.

    Formulas:
    1. Tax Burden = Net Income / EBT
    2. Interest Burden = EBT / EBIT
    3. Operating Margin = EBIT / Revenue
    4. Asset Turnover = Revenue / Total Assets
    5. Equity Multiplier = Total Assets / Total Equity
    ROE = Tax Burden * Interest Burden * Operating Margin * Asset Turnover * Equity Multiplier

    Args:
        profile: FinancialProfile object.

    Returns:
        DuPont5Stage dataclass.
    """
    bs = profile.balance_sheet
    inc = profile.income_stmt

    revenue = profile.get_latest(inc, ["Total Revenue", "Operating Revenue", "Revenue"])
    ebit = profile.get_latest(inc, ["EBIT", "Operating Income", "Operating Profit"])
    ebt = profile.get_latest(inc, ["Pretax Income", "Pre Tax Income", "Income Before Tax"])
    if ebt == 0.0:
        # Fallback: estimate EBT = EBIT - Interest Expense
        interest_exp = abs(profile.get_latest(inc, ["Interest Expense", "Finance Costs"]))
        ebt = ebit - interest_exp

    net_income = profile.get_latest(inc, ["Net Income Common Stockholders", "Net Income"])
    total_assets = profile.get_latest(bs, ["Total Assets"])
    total_equity = profile.get_latest(bs, ["Stockholders Equity", "Total Stockholder Equity", "Common Stock Equity", "Total Equity"])

    tax_burden = (net_income / ebt) if ebt != 0 else 0.0
    interest_burden = (ebt / ebit) if ebit != 0 else 0.0
    op_margin = (ebit / revenue) if revenue > 0 else 0.0
    at = (revenue / total_assets) if total_assets > 0 else 0.0
    em = (total_assets / total_equity) if total_equity > 0 else 0.0

    calc_roe = tax_burden * interest_burden * op_margin * at * em
    actual_roe = (net_income / total_equity) if total_equity > 0 else 0.0
    discrepancy = abs(calc_roe - actual_roe)

    # Narrative diagnosis
    drivers = []
    if interest_burden < 0.70:
        drivers.append("Heavy interest load (EBT/EBIT < 0.70 consumes > 30% of operating profit)")
    else:
        drivers.append("Mild interest drag (EBT/EBIT >= 0.70)")

    if op_margin > 0.18:
        drivers.append("Robust operating margins (> 18%)")
    else:
        drivers.append("Moderate/slim operating margins (<= 18%)")

    interpretation = "; ".join(drivers)

    return DuPont5Stage(
        symbol=profile.symbol,
        tax_burden=tax_burden,
        interest_burden=interest_burden,
        operating_margin=op_margin,
        asset_turnover=at,
        equity_multiplier=em,
        calculated_roe=calc_roe,
        actual_roe=actual_roe,
        discrepancy=discrepancy,
        interpretation=interpretation,
    )
