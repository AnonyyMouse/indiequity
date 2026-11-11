"""
Unit tests for 3-stage and 5-stage DuPont decomposition in indiequity.dupont.
"""

import pytest
from indiequity.data import FinancialProfile
from indiequity.dupont import decompose_dupont_3stage, decompose_dupont_5stage


def test_dupont_3stage(mock_indian_smallcap: FinancialProfile):
    """Verify 3-stage DuPont mathematically reconstructs actual ROE."""
    d3 = decompose_dupont_3stage(mock_indian_smallcap)

    assert d3.symbol == "BHARATPRE.NS"
    assert pytest.approx(d3.net_profit_margin, rel=1e-3) == 0.131
    assert pytest.approx(d3.asset_turnover, rel=1e-3) == 1.200
    assert pytest.approx(d3.equity_multiplier, rel=1e-3) == 1.6667

    # Identity check: NPM * AT * EM == ROE
    assert pytest.approx(d3.calculated_roe, rel=1e-3) == 0.262
    assert pytest.approx(d3.actual_roe, rel=1e-3) == 0.262
    assert d3.discrepancy < 1e-4
    assert len(d3.interpretation) > 0


def test_dupont_5stage(mock_indian_smallcap: FinancialProfile):
    """Verify 5-stage DuPont mathematically reconstructs actual ROE."""
    d5 = decompose_dupont_5stage(mock_indian_smallcap)

    assert d5.symbol == "BHARATPRE.NS"
    # Tax Burden = Net Income / EBT = 157.2 / 210 = 0.74857
    assert pytest.approx(d5.tax_burden, rel=1e-3) == 0.74857
    # Interest Burden = EBT / EBIT = 210 / 240 = 0.875
    assert pytest.approx(d5.interest_burden, rel=1e-3) == 0.875
    # Operating Margin = EBIT / Revenue = 240 / 1200 = 0.20
    assert pytest.approx(d5.operating_margin, rel=1e-3) == 0.200
    # Asset Turnover = Revenue / Total Assets = 1200 / 1000 = 1.20
    assert pytest.approx(d5.asset_turnover, rel=1e-3) == 1.200
    # Equity Multiplier = Total Assets / Total Equity = 1000 / 600 = 1.6667
    assert pytest.approx(d5.equity_multiplier, rel=1e-3) == 1.6667

    # Identity check: TaxBurden * InterestBurden * OperatingMargin * AssetTurnover * EquityMultiplier == ROE
    assert pytest.approx(d5.calculated_roe, rel=1e-3) == 0.262
    assert pytest.approx(d5.actual_roe, rel=1e-3) == 0.262
    assert d5.discrepancy < 1e-4
    assert len(d5.interpretation) > 0
