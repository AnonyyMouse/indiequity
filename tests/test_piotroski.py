"""
Unit tests for the Piotroski F-Score in indiequity.piotroski.

Uses offline two-period mock statements:
1. 'StrongCo' - passes all nine signals (F-Score 9)
2. 'WeakCo'   - fails all nine signals (F-Score 0)
"""

import pandas as pd
import pytest
from indiequity.data import FinancialProfile
from indiequity.piotroski import calculate_piotroski_score


def _make_profile(income_data, balance_data, cashflow_data, symbol, name):
    """Build a FinancialProfile from row-label -> [latest, prior] mappings."""
    cols = ["2024-03-31", "2023-03-31"]
    income_stmt = pd.DataFrame(income_data, index=cols).T
    balance_sheet = pd.DataFrame(balance_data, index=cols).T
    cash_flow = pd.DataFrame(cashflow_data, index=cols).T
    return FinancialProfile(
        symbol=symbol,
        company_name=name,
        sector="Industrials",
        industry="Auto Components",
        current_price=300.0,
        market_cap=15_000_000_000.0,
        shares_outstanding=50_000_000.0,
        beta=1.0,
        income_stmt=income_stmt,
        balance_sheet=balance_sheet,
        cash_flow=cash_flow,
        latest_period_date="2024-03-31",
        available_periods=cols,
        raw_info={},
    )


@pytest.fixture
def strong_company() -> FinancialProfile:
    """Financially strong company improving on every Piotroski dimension."""
    income_data = {
        "Total Revenue": [13_750_000_000.0, 12_000_000_000.0],
        "Gross Profit": [5_500_000_000.0, 4_560_000_000.0],
        "Net Income Common Stockholders": [1_800_000_000.0, 1_500_000_000.0],
    }
    balance_data = {
        "Total Assets": [11_000_000_000.0, 10_000_000_000.0],
        "Current Assets": [4_400_000_000.0, 4_000_000_000.0],
        "Current Liabilities": [2_000_000_000.0, 2_000_000_000.0],
        "Long Term Debt": [1_100_000_000.0, 1_500_000_000.0],
        "Stockholders Equity": [6_500_000_000.0, 6_000_000_000.0],
        "Ordinary Shares Number": [50_000_000.0, 50_000_000.0],
    }
    cashflow_data = {
        "Operating Cash Flow": [2_200_000_000.0, 1_900_000_000.0],
        "Capital Expenditure": [-700_000_000.0, -650_000_000.0],
    }
    return _make_profile(income_data, balance_data, cashflow_data, "STRONG.NS", "StrongCo Ltd")


@pytest.fixture
def weak_company() -> FinancialProfile:
    """Deteriorating company failing every Piotroski signal."""
    income_data = {
        "Total Revenue": [10_800_000_000.0, 12_000_000_000.0],
        "Gross Profit": [3_240_000_000.0, 4_560_000_000.0],
        "Net Income Common Stockholders": [-200_000_000.0, 1_500_000_000.0],
    }
    balance_data = {
        "Total Assets": [12_000_000_000.0, 10_000_000_000.0],
        "Current Assets": [3_000_000_000.0, 4_000_000_000.0],
        "Current Liabilities": [3_000_000_000.0, 2_000_000_000.0],
        "Long Term Debt": [3_000_000_000.0, 1_500_000_000.0],
        "Stockholders Equity": [5_500_000_000.0, 6_000_000_000.0],
        "Ordinary Shares Number": [60_000_000.0, 50_000_000.0],
    }
    cashflow_data = {
        "Operating Cash Flow": [-500_000_000.0, 1_900_000_000.0],
        "Capital Expenditure": [-700_000_000.0, -650_000_000.0],
    }
    return _make_profile(income_data, balance_data, cashflow_data, "WEAK.NS", "WeakCo Ltd")


def test_piotroski_strong_company_scores_nine(strong_company: FinancialProfile):
    """A company improving on all dimensions must earn the full 9 points."""
    result = calculate_piotroski_score(strong_company)

    assert result.symbol == "STRONG.NS"
    assert result.score == 9

    # Profitability
    assert result.roa_positive is True
    assert result.cfo_positive is True
    assert result.roa_improved is True
    assert result.accrual_quality is True

    # Leverage / liquidity
    assert result.leverage_decreased is True
    assert result.liquidity_improved is True
    assert result.no_dilution is True

    # Operating efficiency
    assert result.gross_margin_improved is True
    assert result.asset_turnover_improved is True

    # Underlying metrics: ROA 1.8/11 vs 1.5/10, turnover 13.75/11 vs 12/10
    assert pytest.approx(result.roa, rel=1e-3) == 0.16364
    assert pytest.approx(result.prev_roa, rel=1e-3) == 0.15
    assert pytest.approx(result.asset_turnover, rel=1e-3) == 1.25
    assert pytest.approx(result.prev_asset_turnover, rel=1e-3) == 1.20
    assert pytest.approx(result.gross_margin, rel=1e-3) == 0.40
    assert pytest.approx(result.prev_gross_margin, rel=1e-3) == 0.38
    assert pytest.approx(result.current_ratio, rel=1e-3) == 2.20
    assert pytest.approx(result.prev_current_ratio, rel=1e-3) == 2.00
    assert "Strong" in result.interpretation


def test_piotroski_weak_company_scores_zero(weak_company: FinancialProfile):
    """A deteriorating, loss-making, diluting company must score 0."""
    result = calculate_piotroski_score(weak_company)

    assert result.symbol == "WEAK.NS"
    assert result.score == 0

    assert result.roa_positive is False        # Net income is negative
    assert result.cfo_positive is False        # Operating cash flow is negative
    assert result.roa_improved is False        # ROA fell from +0.15 to negative
    assert result.accrual_quality is False     # CFO (-0.5B) below net income (-0.2B)
    assert result.leverage_decreased is False  # LT debt ratio rose 0.15 -> 0.25
    assert result.liquidity_improved is False  # Current ratio fell 2.0 -> 1.0
    assert result.no_dilution is False         # Shares rose 50M -> 60M
    assert result.gross_margin_improved is False   # Margin fell 38% -> 30%
    assert result.asset_turnover_improved is False  # Turnover fell 1.2 -> 0.9
    assert "Distressed" in result.interpretation


def test_piotroski_single_period_profile_does_not_crash(mock_indian_smallcap: FinancialProfile):
    """A profile with only one period of statements still produces a score."""
    result = calculate_piotroski_score(mock_indian_smallcap)

    assert result.symbol == "BHARATPRE.NS"
    assert 0 <= result.score <= 9
    # Verifiable signals from the single-period fixture:
    assert result.roa_positive is True    # Net income 1.572B on 10B assets
    assert result.cfo_positive is True    # Operating cash flow 2.2B
    assert result.accrual_quality is True  # CFO 2.2B > net income 1.572B
    # Prior-year share count is unavailable, so dilution cannot pass
    assert result.no_dilution is False
