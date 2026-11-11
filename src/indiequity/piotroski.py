"""
Piotroski F-Score Engine (9-point financial health score).

Implements Joseph Piotroski's F-Score (2000): nine binary signals grouped
into profitability, leverage/liquidity, and operating efficiency. Each
signal scores 1 (pass) or 0 (fail), giving a composite score from 0
(weakest) to 9 (strongest) for the latest fiscal year versus the prior year.
"""

from dataclasses import dataclass
from indiequity.data import FinancialProfile


@dataclass
class PiotroskiScore:
    """
    Piotroski F-Score breakdown for a single company.

    Signals (each True = 1 point):
    Profitability:
      1. roa_positive        - Return on Assets > 0
      2. cfo_positive        - Operating Cash Flow > 0
      3. roa_improved        - ROA increased vs prior year (ΔROA > 0)
      4. accrual_quality     - CFO exceeds Net Income (low accruals)
    Leverage, Liquidity & Source of Funds:
      5. leverage_decreased  - Long-term debt ratio fell vs prior year
      6. liquidity_improved  - Current ratio increased vs prior year
      7. no_dilution         - No new shares issued vs prior year
    Operating Efficiency:
      8. gross_margin_improved - Gross margin increased vs prior year
      9. asset_turnover_improved - Asset turnover increased vs prior year
    """
    symbol: str

    # Profitability signals
    roa_positive: bool
    cfo_positive: bool
    roa_improved: bool
    accrual_quality: bool

    # Leverage / liquidity signals
    leverage_decreased: bool
    liquidity_improved: bool
    no_dilution: bool

    # Operating efficiency signals
    gross_margin_improved: bool
    asset_turnover_improved: bool

    # Underlying metrics (latest year, prior year)
    roa: float
    prev_roa: float
    leverage_ratio: float
    prev_leverage_ratio: float
    current_ratio: float
    prev_current_ratio: float
    gross_margin: float
    prev_gross_margin: float
    asset_turnover: float
    prev_asset_turnover: float

    score: int
    interpretation: str


def _ratio(numerator: float, denominator: float) -> float:
    """Safe division returning 0.0 when the denominator is not positive."""
    return (numerator / denominator) if denominator > 0 else 0.0


def calculate_piotroski_score(profile: FinancialProfile) -> PiotroskiScore:
    """
    Compute the 9-point Piotroski F-Score for the latest fiscal year.

    All year-over-year signals compare the latest period (period_idx=0)
    against the prior period (period_idx=1) of the company's statements.

    Args:
        profile: FinancialProfile object.

    Returns:
        PiotroskiScore dataclass with the nine signals, the underlying
        metrics for both periods, the composite score (0-9), and a
        narrative interpretation.
    """
    bs = profile.balance_sheet
    inc = profile.income_stmt
    cf = profile.cash_flow

    # Latest year (period 0)
    net_income = profile.get_latest(inc, ["Net Income Common Stockholders", "Net Income"], default=0.0)
    revenue = profile.get_latest(inc, ["Total Revenue", "Operating Revenue", "Revenue"], default=0.0)
    gross_profit = profile.get_latest(inc, ["Gross Profit"], default=0.0)
    cfo = profile.get_latest(cf, [
        "Operating Cash Flow",
        "Cash Flow From Continuing Operating Activities",
    ], default=0.0)
    total_assets = profile.get_latest(bs, ["Total Assets"], default=0.0)
    lt_debt = profile.get_latest(bs, [
        "Long Term Debt",
        "Long Term Debt Non Current",
        "Long Term Debt And Capital Lease Obligation",
    ], default=0.0)
    current_assets = profile.get_latest(bs, ["Current Assets", "Total Current Assets"], default=0.0)
    current_liabilities = profile.get_latest(bs, ["Current Liabilities", "Total Current Liabilities"], default=0.0)
    shares = profile.get_latest(bs, ["Ordinary Shares Number", "Share Issued"], default=None)
    if shares is None:
        shares = profile.shares_outstanding or 0.0

    # Prior year (period 1)
    prev_net_income = profile.get_latest(inc, ["Net Income Common Stockholders", "Net Income"], default=0.0, period_idx=1)
    prev_revenue = profile.get_latest(inc, ["Total Revenue", "Operating Revenue", "Revenue"], default=0.0, period_idx=1)
    prev_gross_profit = profile.get_latest(inc, ["Gross Profit"], default=0.0, period_idx=1)
    prev_total_assets = profile.get_latest(bs, ["Total Assets"], default=0.0, period_idx=1)
    prev_lt_debt = profile.get_latest(bs, [
        "Long Term Debt",
        "Long Term Debt Non Current",
        "Long Term Debt And Capital Lease Obligation",
    ], default=0.0, period_idx=1)
    prev_current_assets = profile.get_latest(bs, ["Current Assets", "Total Current Assets"], default=0.0, period_idx=1)
    prev_current_liabilities = profile.get_latest(bs, ["Current Liabilities", "Total Current Liabilities"], default=0.0, period_idx=1)
    prev_shares = profile.get_latest(bs, ["Ordinary Shares Number", "Share Issued"], default=None, period_idx=1)

    # Underlying metrics
    roa = _ratio(net_income, total_assets)
    prev_roa = _ratio(prev_net_income, prev_total_assets)
    leverage_ratio = _ratio(lt_debt, total_assets)
    prev_leverage_ratio = _ratio(prev_lt_debt, prev_total_assets)
    current_ratio = _ratio(current_assets, current_liabilities)
    prev_current_ratio = _ratio(prev_current_assets, prev_current_liabilities)
    gross_margin = _ratio(gross_profit, revenue)
    prev_gross_margin = _ratio(prev_gross_profit, prev_revenue)
    asset_turnover = _ratio(revenue, total_assets)
    prev_asset_turnover = _ratio(prev_revenue, prev_total_assets)

    # Profitability signals
    roa_positive = roa > 0
    cfo_positive = cfo > 0
    roa_improved = roa > prev_roa
    # Accrual signal: CFO / Total Assets > ROA, i.e. CFO > Net Income
    accrual_quality = cfo > net_income

    # Leverage / liquidity signals
    leverage_decreased = leverage_ratio < prev_leverage_ratio
    liquidity_improved = current_ratio > prev_current_ratio
    # No dilution: share count did not rise. If the prior-year share count is
    # unavailable the signal cannot be verified and scores 0.
    no_dilution = prev_shares is not None and shares <= prev_shares

    # Operating efficiency signals
    gross_margin_improved = gross_margin > prev_gross_margin
    asset_turnover_improved = asset_turnover > prev_asset_turnover

    signals = [
        roa_positive, cfo_positive, roa_improved, accrual_quality,
        leverage_decreased, liquidity_improved, no_dilution,
        gross_margin_improved, asset_turnover_improved,
    ]
    score = sum(1 for s in signals if s)

    # Narrative diagnosis
    if score >= 8:
        strength = "Strong financial health (F-Score 8-9: high-quality balance sheet and improving fundamentals)"
    elif score >= 6:
        strength = "Sound financial health (F-Score 6-7: most Piotroski signals pass)"
    elif score >= 4:
        strength = "Mixed financial health (F-Score 4-5: fundamentals neither clearly improving nor deteriorating)"
    elif score >= 2:
        strength = "Weak financial health (F-Score 2-3: several profitability or leverage signals fail)"
    else:
        strength = "Distressed financial profile (F-Score 0-1: fails nearly all Piotroski signals)"

    failed = []
    if not roa_positive:
        failed.append("negative ROA")
    if not cfo_positive:
        failed.append("negative operating cash flow")
    if not roa_improved:
        failed.append("ROA not improving")
    if not accrual_quality:
        failed.append("earnings not backed by cash flow (accruals)")
    if not leverage_decreased:
        failed.append("long-term leverage not reduced")
    if not liquidity_improved:
        failed.append("current ratio not improving")
    if not no_dilution:
        failed.append("share count increased or unverifiable (dilution)")
    if not gross_margin_improved:
        failed.append("gross margin not improving")
    if not asset_turnover_improved:
        failed.append("asset turnover not improving")

    interpretation = strength
    if failed:
        interpretation += "; Watch: " + ", ".join(failed)

    return PiotroskiScore(
        symbol=profile.symbol,
        roa_positive=roa_positive,
        cfo_positive=cfo_positive,
        roa_improved=roa_improved,
        accrual_quality=accrual_quality,
        leverage_decreased=leverage_decreased,
        liquidity_improved=liquidity_improved,
        no_dilution=no_dilution,
        gross_margin_improved=gross_margin_improved,
        asset_turnover_improved=asset_turnover_improved,
        roa=roa,
        prev_roa=prev_roa,
        leverage_ratio=leverage_ratio,
        prev_leverage_ratio=prev_leverage_ratio,
        current_ratio=current_ratio,
        prev_current_ratio=prev_current_ratio,
        gross_margin=gross_margin,
        prev_gross_margin=prev_gross_margin,
        asset_turnover=asset_turnover,
        prev_asset_turnover=prev_asset_turnover,
        score=score,
        interpretation=interpretation,
    )
