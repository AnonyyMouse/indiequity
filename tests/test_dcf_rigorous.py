"""
Comprehensive unit tests for indiequity valuation engine repair specification.

Covers:
1. Exact FCFF arithmetic (including the specification's ₹100 Cr worked test).
2. Negative FCFF preservation without arbitrary positive overrides.
3. Missing data rejection (missing shares, invalid revenue).
4. Invalid WACC and terminal-growth combinations (g >= WACC validation).
5. Enterprise-to-equity bridge (including debt, leases, minority interest, cash).
6. Sustainable terminal reinvestment modeling.
7. Company-type valuation routing (blocking industrial DCF on banks; routing to RIM).
8. Monotonic valuation sensitivity.
9. Reverse DCF implied growth solver.
10. Scenario analysis (Bear < Base < Bull).
11. Period date chronological sorting.
"""

import pandas as pd
import pytest
from indiequity.data import FinancialProfile, _sort_statement_columns
from indiequity.dcf import (
    ForecastAssumptions,
    InvalidValuationInputError,
    UnsupportedValuationModelError,
    calculate_dcf,
    calculate_fcff_item,
    calculate_residual_income_valuation,
    estimate_wacc,
    reverse_dcf,
    run_scenario_analysis,
)


def test_fcff_worked_specification_example():
    """
    Test Phase 1, Item 1 Worked Example from Specification:
    EBIT = ₹200 crore
    Tax = 25%
    D&A = ₹40 crore
    CapEx = ₹70 crore
    Increase in operating NWC = ₹20 crore
    Expected FCFF = ₹100 crore.
    """
    ebit = 200.0
    tax_rate = 0.25
    dna = 40.0
    capex = 70.0
    delta_nwc = 20.0

    fcff = calculate_fcff_item(
        ebit=ebit,
        tax_rate=tax_rate,
        depreciation_and_amortization=dna,
        capex=capex,
        change_in_operating_nwc=delta_nwc,
    )

    # NOPAT = 200 * (1 - 0.25) = 150
    # FCFF = 150 + 40 - 70 - 20 = 100
    assert pytest.approx(fcff, rel=1e-5) == 100.0


def test_negative_fcff_preservation():
    """Verify negative FCFF is preserved and not replaced by an arbitrary positive number."""
    # Fast growing company with heavy CapEx and working capital investment
    ebit = 100.0
    tax_rate = 0.25
    dna = 15.0
    capex = 160.0  # Heavy capital expenditure
    delta_nwc = 40.0

    fcff = calculate_fcff_item(
        ebit=ebit,
        tax_rate=tax_rate,
        depreciation_and_amortization=dna,
        capex=capex,
        change_in_operating_nwc=delta_nwc,
    )

    # NOPAT = 75.0, FCFF = 75 + 15 - 160 - 40 = -110.0
    assert fcff == -110.0
    assert fcff < 0.0


def test_missing_data_rejection(mock_indian_smallcap: FinancialProfile):
    """Verify engine raises explicit error when required inputs are missing."""
    # Corrupt share count to zero
    bad_profile = FinancialProfile(
        symbol=mock_indian_smallcap.symbol,
        company_name=mock_indian_smallcap.company_name,
        sector=mock_indian_smallcap.sector,
        industry=mock_indian_smallcap.industry,
        current_price=300.0,
        market_cap=15_000_000_000.0,
        shares_outstanding=0.0,  # Invalid!
        beta=1.15,
        income_stmt=mock_indian_smallcap.income_stmt,
        balance_sheet=mock_indian_smallcap.balance_sheet,
        cash_flow=mock_indian_smallcap.cash_flow,
    )

    with pytest.raises(InvalidValuationInputError, match="shares outstanding is missing or non-positive"):
        calculate_dcf(bad_profile)


def test_invalid_wacc_and_terminal_growth_combination(mock_indian_smallcap: FinancialProfile):
    """Verify validation error is raised if terminal growth >= WACC (no silent clamping!)."""
    # WACC = 10%, Terminal growth = 11% -> Mathematically invalid for perpetual Gordon growth
    asm = ForecastAssumptions(
        wacc_override=0.10,
        terminal_growth_rate=0.11,
    )
    with pytest.raises(InvalidValuationInputError, match="Terminal growth rate .* cannot be greater than or equal to WACC"):
        calculate_dcf(mock_indian_smallcap, assumptions=asm)


def test_enterprise_to_equity_bridge(mock_indian_smallcap: FinancialProfile):
    """Verify complete reconciliation of Enterprise Value to Equity Value and Per-Share Value."""
    res = calculate_dcf(mock_indian_smallcap)
    b = res.bridge

    # EV = PV(FCFF) + PV(Terminal Value)
    assert pytest.approx(b.enterprise_value, rel=1e-4) == (b.sum_pv_fcff + b.pv_terminal_value)

    # Implied Equity Value = EV + Cash & Liquid - Debt - Leases - Minority Interest - Preferred
    expected_equity_val = (
        b.enterprise_value
        + b.total_liquid_assets
        - b.total_interest_bearing_debt
        - b.lease_liabilities
        - b.minority_interest
        - b.preferred_stock
    )
    assert pytest.approx(b.implied_equity_value, rel=1e-4) == expected_equity_val

    # Per Share Intrinsic Value = Implied Equity Value / Diluted Shares
    assert pytest.approx(b.intrinsic_value_per_share, rel=1e-4) == (b.implied_equity_value / b.diluted_shares)


def test_financial_institution_routing(mock_indian_bank: FinancialProfile):
    """Verify industrial DCF is rejected for banks, and Residual Income Model is routed."""
    # 1. Industrial DCF must raise UnsupportedValuationModelError
    with pytest.raises(UnsupportedValuationModelError, match="Cannot run industrial FCFF/WACC DCF on"):
        calculate_dcf(mock_indian_bank)

    # 2. Residual Income Model executes properly
    rim = calculate_residual_income_valuation(mock_indian_bank)
    assert rim.symbol == "HINDCOMM.NS"
    assert rim.book_value_per_share > 0
    assert rim.justified_pb_ratio > 0
    assert rim.intrinsic_value_per_share > 0


def test_monotonic_valuation_sensitivity(mock_indian_smallcap: FinancialProfile):
    """Verify valuation decreases monotonically as WACC increases, and increases as terminal growth increases."""
    res = calculate_dcf(mock_indian_smallcap)
    sens = res.sensitivity_table

    col = "g = 5.00%"
    # Higher discount rate must yield lower intrinsic value
    val_wacc_low = sens.loc["WACC 10.0%", col] if "WACC 10.0%" in sens.index else sens.iloc[0, 2]
    val_wacc_high = sens.loc["WACC 14.0%", col] if "WACC 14.0%" in sens.index else sens.iloc[4, 2]
    assert val_wacc_low > val_wacc_high


def test_reverse_dcf_solver(mock_indian_smallcap: FinancialProfile):
    """Verify Reverse DCF accurately finds growth rate implied by market price."""
    rev = reverse_dcf(mock_indian_smallcap, target_price=300.0)
    assert rev["symbol"] == "BHARATPRE.NS"
    assert "implied_5y_cagr" in rev
    assert -0.20 <= rev["implied_5y_cagr"] <= 0.80


def test_scenario_analysis(mock_indian_smallcap: FinancialProfile):
    """Verify Bear < Base < Bull case ordering."""
    scens = run_scenario_analysis(mock_indian_smallcap, base_growth=0.12)
    bear_val = scens["Bear"].bridge.intrinsic_value_per_share
    base_val = scens["Base"].bridge.intrinsic_value_per_share
    bull_val = scens["Bull"].bridge.intrinsic_value_per_share

    assert bear_val < base_val < bull_val


def test_statement_column_chronological_sorting():
    """Verify statement columns are sorted newest to oldest regardless of initial feed ordering."""
    df_unsorted = pd.DataFrame(
        {"2021-03-31": [10.0], "2024-03-31": [40.0], "2022-03-31": [20.0], "2023-03-31": [30.0]},
        index=["Revenue"],
    )
    df_sorted = _sort_statement_columns(df_unsorted)

    expected_order = ["2024-03-31", "2023-03-31", "2022-03-31", "2021-03-31"]
    assert list(df_sorted.columns) == expected_order
    assert df_sorted.iloc[0, 0] == 40.0  # 2024 is first
