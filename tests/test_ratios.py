"""
Unit tests for financial ratios calculation in indiequity.ratios.
"""

import pytest
from indiequity.data import FinancialProfile
from indiequity.ratios import calculate_ratios


def test_calculate_ratios(mock_indian_smallcap: FinancialProfile):
    """Verify calculated financial metrics match accounting hand-calculations."""
    ratios = calculate_ratios(mock_indian_smallcap)

    # Basic company identity
    assert ratios.symbol == "BHARATPRE.NS"
    assert ratios.company_name == "Bharat Precision Ltd"

    # 1. Return on Equity (ROE): Net Income / Total Equity = 157.2 Cr / 600 Cr = 26.2%
    assert ratios.roe is not None
    assert pytest.approx(ratios.roe, rel=1e-3) == 0.262

    # 2. Return on Capital Employed (ROCE):
    # EBIT / (Total Assets - Current Liabilities) = 240 Cr / (1000 Cr - 200 Cr) = 240 / 800 = 30.0%
    assert ratios.roce is not None
    assert pytest.approx(ratios.roce, rel=1e-3) == 0.300

    # 3. Debt to Equity (D/E): Total Debt / Total Equity = 200 Cr / 600 Cr = 0.333
    assert ratios.debt_to_equity is not None
    assert pytest.approx(ratios.debt_to_equity, rel=1e-3) == 0.3333

    # 4. Interest Coverage Ratio: EBIT / Interest Expense = 240 Cr / 30 Cr = 8.0x
    assert ratios.interest_coverage is not None
    assert pytest.approx(ratios.interest_coverage, rel=1e-3) == 8.0

    # 5. Free Cash Flow (FCF) & FCF Yield:
    # FCF = CFO - CapEx = 220 Cr - 70 Cr = 150 Cr (1.5 billion INR)
    # FCF Yield = FCF / Market Cap = 150 Cr / 1,500 Cr = 10.0%
    assert pytest.approx(ratios.free_cash_flow, rel=1e-3) == 1_500_000_000.0
    assert ratios.fcf_yield is not None
    assert pytest.approx(ratios.fcf_yield, rel=1e-3) == 0.100

    # 6. EV / EBITDA:
    # EV = 16,500 Cr, EBITDA = 240 + 40 = 280 Cr -> EV/EBITDA = 16,500 / 2,800 = 5.8928
    assert ratios.ev_to_ebitda is not None
    assert pytest.approx(ratios.ev_to_ebitda, rel=1e-3) == 5.8928

    # 7. Margins & Turnover
    assert pytest.approx(ratios.operating_margin, rel=1e-3) == 0.200  # 240 / 1200
    assert pytest.approx(ratios.net_profit_margin, rel=1e-3) == 0.131  # 157.2 / 1200
    assert pytest.approx(ratios.asset_turnover, rel=1e-3) == 1.200  # 1200 / 1000
