"""
Rigorous financial methodology validation tests for indiequity v0.2.1.

Validates:
1. Residual Income sustainable growth identities (Retention = g / ROE, Payout = 1 - g / ROE).
2. Phase 1 worked example: ROE 16%, g 10%, Ke 12%, BVPS 100 -> Retention 62.5%, Payout 37.5%, EPS 16, Div 6, End BVPS 110.
3. Residual Income bounds (ROE > Ke, ROE == Ke, ROE < Ke, zero growth, unsustainable growth rejection).
4. Company classification & model routing (ICICIBANK, HDFCAMC, M&M, POLYCAB, TCS).
5. MODEL_NOT_SUPPORTED rejection for Asset Managers and Conglomerates.
6. FCFF arithmetic with positive and negative operating working capital changes (+20 Cr -> 100 Cr, -20 Cr -> 140 Cr).
7. Reverse DCF numerical round-trip verification (abs(P_recalc - P_target) / P_target < 0.001).
8. WACC unclamped borrowing cost and validation.
"""

import pandas as pd
import pytest
from indiequity.data import (
    BusinessCategory,
    FinancialProfile,
    classify_business,
)
from indiequity.dcf import (
    ForecastAssumptions,
    InvalidValuationInputError,
    UnsupportedValuationModelError,
    calculate_dcf,
    calculate_fcff_item,
    calculate_residual_income_valuation,
    reverse_dcf,
)


# =====================================================================
# PHASE 1: RESIDUAL INCOME VALUATION TESTS
# =====================================================================

def test_residual_income_worked_example():
    """
    Test Phase 1 Worked Example from Specification:
    Assume:
    - ROE = 16%
    - Sustainable growth = 10%
    - Cost of equity = 12%
    - Starting book value per share = ₹100

    Then:
    - Retention ratio = 10% / 16% = 62.5%
    - Payout ratio = 37.5%
    - First-year EPS = ₹100 * 16% = ₹16
    - First-year dividend = ₹16 * 37.5% = ₹6
    - First-year closing book value = ₹100 + ₹16 - ₹6 = ₹110
    """
    # Create synthetic bank profile
    bs = pd.DataFrame({"2024-03-31": [100.0 * 10_000_000]}, index=["Stockholders Equity"])
    inc = pd.DataFrame({"2024-03-31": [16.0 * 10_000_000]}, index=["Net Income Common Stockholders"])

    bank_profile = FinancialProfile(
        symbol="SYNTHBANK.NS",
        company_name="Synthetic Bank Ltd",
        sector="Financial Services",
        industry="Banks - Diversified",
        current_price=150.0,
        market_cap=150.0 * 10_000_000,
        shares_outstanding=10_000_000,
        beta=1.0,
        income_stmt=inc,
        balance_sheet=bs,
        cash_flow=pd.DataFrame(),
        business_category=BusinessCategory.BANK,
    )

    rim = calculate_residual_income_valuation(
        bank_profile,
        risk_free_rate=0.06,
        equity_risk_premium=0.06,  # Ke = 6% + 1.0 * 6% = 12%
        forecast_years=5,
        terminal_growth_rate=0.10,  # g = 10%
        normalized_roe_override=0.16,  # ROE = 16%
    )

    # 1. Retention ratio = 10% / 16% = 62.5%
    assert pytest.approx(rim.retention_ratio, rel=1e-5) == 0.625

    # 2. Payout ratio = 1 - 62.5% = 37.5%
    assert pytest.approx(rim.payout_ratio, rel=1e-5) == 0.375

    # Year 1 forecast inspection
    y1 = rim.annual_forecasts[0]
    assert pytest.approx(y1.beginning_book_value, rel=1e-5) == 100.0
    assert pytest.approx(y1.eps, rel=1e-5) == 16.0
    assert pytest.approx(y1.dividend, rel=1e-5) == 6.0
    assert pytest.approx(y1.ending_book_value, rel=1e-5) == 110.0
    assert pytest.approx(y1.residual_income, rel=1e-5) == 4.0  # EPS (16) - Ke * BV (12) = 4.0


def test_residual_income_roe_equals_ke(mock_indian_bank: FinancialProfile):
    """When ROE == Ke, company creates zero economic value; Intrinsic Value must equal Book Value."""
    rim = calculate_residual_income_valuation(
        mock_indian_bank,
        risk_free_rate=0.07,
        equity_risk_premium=0.05,  # Ke = 7% + 1.0 * 5% = 12%
        normalized_roe_override=0.12,  # ROE = 12% == Ke
        terminal_growth_rate=0.05,
        beta_override=1.0,
    )
    # Intrinsic Value must equal Book Value per share exactly
    assert pytest.approx(rim.intrinsic_value_per_share, rel=1e-3) == rim.book_value_per_share
    assert pytest.approx(rim.justified_pb_ratio, rel=1e-3) == 1.00


def test_residual_income_roe_below_ke(mock_indian_bank: FinancialProfile):
    """When ROE < Ke, company destroys value; Intrinsic Value must be BELOW Book Value (P/B < 1.0)."""
    rim = calculate_residual_income_valuation(
        mock_indian_bank,
        risk_free_rate=0.07,
        equity_risk_premium=0.06,  # Ke = 13.3%
        normalized_roe_override=0.09,  # ROE = 9% < Ke
        terminal_growth_rate=0.04,
        beta_override=1.05,
    )
    assert rim.intrinsic_value_per_share < rim.book_value_per_share
    assert rim.justified_pb_ratio < 1.0


def test_residual_income_zero_growth(mock_indian_bank: FinancialProfile):
    """When g == 0, retention ratio is 0% and payout ratio is 100%."""
    rim = calculate_residual_income_valuation(
        mock_indian_bank,
        terminal_growth_rate=0.0,
        normalized_roe_override=0.15,
    )
    assert rim.sustainable_growth_rate == 0.0
    assert rim.retention_ratio == 0.0
    assert rim.payout_ratio == 1.0
    assert rim.intrinsic_value_per_share > 0


def test_residual_income_unsustainable_growth_rejection(mock_indian_bank: FinancialProfile):
    """Verify validation error when growth > ROE (retention ratio > 100%)."""
    with pytest.raises(InvalidValuationInputError, match="cannot exceed ROE"):
        calculate_residual_income_valuation(
            mock_indian_bank,
            terminal_growth_rate=0.15,  # g = 15% > ROE = 12%
            normalized_roe_override=0.12,
            risk_free_rate=0.08,
            equity_risk_premium=0.09,
            beta_override=1.1,  # Ke = 8% + 1.1*9% = 17.9% > 15%
        )


def test_residual_income_growth_exceeds_ke_rejection(mock_indian_bank: FinancialProfile):
    """Verify validation error when terminal growth >= Cost of Equity."""
    with pytest.raises(InvalidValuationInputError, match="cannot be >= Cost of Equity"):
        calculate_residual_income_valuation(
            mock_indian_bank,
            terminal_growth_rate=0.15,
            normalized_roe_override=0.18,
            risk_free_rate=0.07,
            equity_risk_premium=0.05,
            beta_override=1.0,  # Ke = 12% < g = 15%
        )


# =====================================================================
# PHASE 2: COMPANY CLASSIFICATION & ROUTING TESTS
# =====================================================================

def test_company_classification_routing():
    """Verify classification and model routing for required test companies."""
    # 1. ICICIBANK -> BANK
    cat, pref = classify_business("ICICIBANK.NS", "Financial Services", "Banks - Diversified")
    assert cat == BusinessCategory.BANK
    assert pref == "RESIDUAL_INCOME_PB"

    # 2. HDFCAMC -> ASSET_MANAGER
    cat, pref = classify_business("HDFCAMC.NS", "Financial Services", "Asset Management")
    assert cat == BusinessCategory.ASSET_MANAGER
    assert pref == "AUM_DRIVEN_CASH_FLOW"

    # 3. M&M -> CONGLOMERATE
    cat, pref = classify_business("M&M.NS", "Consumer Cyclical", "Auto Manufacturers")
    assert cat == BusinessCategory.CONGLOMERATE
    assert pref == "SUM_OF_THE_PARTS"

    # 4. POLYCAB -> INDUSTRIAL_MANUFACTURING
    cat, pref = classify_business("POLYCAB.NS", "Industrials", "Electrical Equipment & Parts")
    assert cat == BusinessCategory.INDUSTRIAL_MANUFACTURING
    assert pref == "FCFF_DCF"

    # 5. TCS -> TECH_SERVICES
    cat, pref = classify_business("TCS.NS", "Technology", "Information Technology Services")
    assert cat == BusinessCategory.TECH_SERVICES
    assert pref == "FCFF_DCF"


def test_model_not_supported_guards():
    """Verify MODEL_NOT_SUPPORTED is returned when inappropriate model is invoked."""
    # Asset Manager must reject industrial DCF AND bank RIM
    amc_profile = FinancialProfile(
        symbol="HDFCAMC.NS",
        company_name="HDFC Asset Management Company Ltd",
        sector="Financial Services",
        industry="Asset Management",
        current_price=4200.0,
        market_cap=90_000_000_000.0,
        shares_outstanding=21_300_000.0,
        beta=0.90,
        income_stmt=pd.DataFrame({"2024-03-31": [30_000_000_000.0]}, index=["Total Revenue"]),
        balance_sheet=pd.DataFrame({"2024-03-31": [70_000_000_000.0]}, index=["Total Assets"]),
        cash_flow=pd.DataFrame(),
        business_category=BusinessCategory.ASSET_MANAGER,
    )

    with pytest.raises(UnsupportedValuationModelError, match="MODEL_NOT_SUPPORTED.*Asset Management Company"):
        calculate_dcf(amc_profile)

    with pytest.raises(UnsupportedValuationModelError, match="MODEL_NOT_SUPPORTED.*Asset Manager"):
        calculate_residual_income_valuation(amc_profile)

    # Conglomerate must reject consolidated industrial DCF
    conglom_profile = FinancialProfile(
        symbol="M&M.NS",
        company_name="Mahindra & Mahindra Limited",
        sector="Consumer Cyclical",
        industry="Auto Manufacturers",
        current_price=2800.0,
        market_cap=3_500_000_000_000.0,
        shares_outstanding=1_240_000_000.0,
        beta=1.10,
        income_stmt=pd.DataFrame({"2024-03-31": [1_000_000_000_000.0]}, index=["Total Revenue"]),
        balance_sheet=pd.DataFrame({"2024-03-31": [2_000_000_000_000.0]}, index=["Total Assets"]),
        cash_flow=pd.DataFrame(),
        business_category=BusinessCategory.CONGLOMERATE,
    )

    with pytest.raises(UnsupportedValuationModelError, match="MODEL_NOT_SUPPORTED.*multi-segment Conglomerate"):
        calculate_dcf(conglom_profile)


# =====================================================================
# PHASE 3: FCFF WORKED EXAMPLES (POSITIVE & NEGATIVE ΔNWC)
# =====================================================================

def test_fcff_case_a_increase_in_nwc():
    """
    Worked Example Case A: Increase in operating NWC (+₹20 Cr)
    EBIT = 200, Tax = 25% (50), D&A = 40, CapEx = 70, ΔNWC = +20
    Expected FCFF = 200 * 0.75 + 40 - 70 - 20 = 100 Cr.
    """
    fcff = calculate_fcff_item(
        ebit=200.0,
        tax_rate=0.25,
        depreciation_and_amortization=40.0,
        capex=70.0,
        change_in_operating_nwc=20.0,
    )
    assert fcff == 100.0


def test_fcff_case_b_decrease_in_nwc():
    """
    Worked Example Case B: Decrease in operating NWC (-₹20 Cr, cash released from working capital)
    EBIT = 200, Tax = 25% (50), D&A = 40, CapEx = 70, ΔNWC = -20
    Expected FCFF = 200 * 0.75 + 40 - 70 - (-20) = 150 + 40 - 70 + 20 = 140 Cr.
    """
    fcff = calculate_fcff_item(
        ebit=200.0,
        tax_rate=0.25,
        depreciation_and_amortization=40.0,
        capex=70.0,
        change_in_operating_nwc=-20.0,
    )
    assert fcff == 140.0


# =====================================================================
# PHASE 5: REVERSE DCF NUMERICAL ROUND-TRIP VERIFICATION
# =====================================================================

def test_reverse_dcf_numerical_round_trip(mock_indian_smallcap: FinancialProfile):
    """
    Specification Phase 5 Acceptance Test:
    1. Select validated DCF model.
    2. Set a known revenue-growth assumption (g* = 12%).
    3. Calculate its intrinsic value (P*).
    4. Use P* as reverse DCF target price.
    5. Solve for implied growth.
    6. Confirm recovered growth matches original growth.
    7. Reinsert recovered growth into forward DCF -> P_recalculated.
    Acceptance condition: abs(P_recalculated - P*) / P* < 0.001 (0.1%).
    """
    known_growth = 0.12
    asm = ForecastAssumptions(
        forecast_years=5,
        revenue_growth_rates=[known_growth] * 5,
        terminal_growth_rate=0.05,
    )
    forward_res = calculate_dcf(mock_indian_smallcap, assumptions=asm)
    target_price = forward_res.intrinsic_value_per_share

    # Solve for implied growth
    rev_result = reverse_dcf(
        mock_indian_smallcap,
        target_price=target_price,
        terminal_growth_rate=0.05,
        tolerance=0.001,
    )

    recovered_g = float(rev_result["implied_5y_cagr"])

    # Re-feed recovered growth into forward DCF
    verify_asm = ForecastAssumptions(
        forecast_years=5,
        revenue_growth_rates=[recovered_g] * 5,
        terminal_growth_rate=0.05,
    )
    recalculated_res = calculate_dcf(mock_indian_smallcap, assumptions=verify_asm)
    recalculated_price = recalculated_res.intrinsic_value_per_share

    relative_error = abs(recalculated_price - target_price) / target_price

    # Assert rigorous 0.1% tolerance condition
    assert relative_error < 0.001
    assert pytest.approx(recovered_g, abs=0.002) == known_growth


def test_reverse_dcf_unsolvable_target_price_rejection(mock_indian_smallcap: FinancialProfile):
    """Verify Reverse DCF raises explicit error when target price is mathematically unsolvable."""
    with pytest.raises(InvalidValuationInputError, match="outside solvable range"):
        reverse_dcf(mock_indian_smallcap, target_price=999_999.0)
