"""
Unit tests for the DCF valuation module in indiequity.dcf.
"""

import pytest
from indiequity.data import FinancialProfile
from indiequity.dcf import calculate_dcf, estimate_wacc


def test_estimate_wacc(mock_indian_smallcap: FinancialProfile):
    """Verify estimated WACC falls within standard Indian market bounds."""
    wacc = estimate_wacc(mock_indian_smallcap)
    # Indian small-cap WACC with Beta ~1.15, Rf 7%, ERP 6.5% should be ~11% - 15%
    assert 0.09 <= wacc <= 0.17


def test_calculate_dcf(mock_indian_smallcap: FinancialProfile):
    """Verify 5-year FCFF DCF model and intrinsic value calculation."""
    dcf = calculate_dcf(
        mock_indian_smallcap,
        forecast_years=5,
        fcf_growth_rate=0.12,
        wacc=0.12,
        terminal_growth_rate=0.05,
    )

    assert dcf.symbol == "BHARATPRE.NS"
    assert dcf.company_name == "Bharat Precision Ltd"
    assert len(dcf.projected_fcff) == 5
    assert len(dcf.pv_projected_fcff) == 5

    # Check cash flow compounding
    assert dcf.projected_fcff[1] > dcf.projected_fcff[0]
    assert dcf.sum_pv_fcff > 0

    # Gordon growth Terminal Value check
    assert dcf.terminal_value > 0
    assert dcf.pv_terminal_value > 0

    # Enterprise Value to Equity Value bridge
    # Implied Equity Value = EV + Cash - Debt
    assert pytest.approx(dcf.implied_equity_value, rel=1e-3) == (
        dcf.enterprise_value + dcf.cash_and_equivalents - dcf.total_debt
    )

    # Intrinsic value per share
    assert dcf.intrinsic_value_per_share > 0
    assert pytest.approx(dcf.intrinsic_value_per_share, rel=1e-3) == (
        dcf.implied_equity_value / dcf.shares_outstanding
    )

    # Verify 2D Sensitivity Table structure
    assert dcf.sensitivity_table.shape == (5, 5)
    assert "g = 5.00%" in dcf.sensitivity_table.columns
    assert "WACC 12.0%" in dcf.sensitivity_table.index

    # Check sensitivity property: Lower WACC should yield HIGHER per share value
    col = "g = 5.00%"
    assert dcf.sensitivity_table.loc["WACC 10.0%", col] > dcf.sensitivity_table.loc["WACC 14.0%", col]
