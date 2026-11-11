"""
Tests for indiequity.config module.
"""

from indiequity.config import (
    CRORE_DIVISOR,
    DEFAULT_EQUITY_RISK_PREMIUM,
    DEFAULT_RISK_FREE_RATE,
    DEFAULT_TAX_RATE,
    DEFAULT_TERMINAL_GROWTH_RATE,
    format_crores,
    format_percentage,
    resolve_ticker,
)


def test_market_constants():
    """Verify Indian benchmark constants are reasonably set."""
    assert 0.05 <= DEFAULT_RISK_FREE_RATE <= 0.09
    assert 0.04 <= DEFAULT_EQUITY_RISK_PREMIUM <= 0.09
    assert 0.03 <= DEFAULT_TERMINAL_GROWTH_RATE <= 0.07
    assert 0.20 <= DEFAULT_TAX_RATE <= 0.35
    assert CRORE_DIVISOR == 10_000_000.0


def test_format_crores():
    """Test rupee to crore formatting with Indian comma style."""
    # 1,250.50 Crores
    raw_amount = 12_505_000_000
    assert format_crores(raw_amount) == "₹1,250.50 Cr"

    # Small amount: 45 Lakhs = 0.45 Crore
    assert format_crores(4_500_000) == "₹0.45 Cr"

    # Exact 1 Crore
    assert format_crores(10_000_000) == "₹1.00 Cr"

    # Negative amount (-50 Crores)
    assert format_crores(-500_000_000) == "-₹50.00 Cr"

    # None handling
    assert format_crores(None) == "N/A"


def test_format_percentage():
    """Test decimal to percentage formatting."""
    assert format_percentage(0.1852) == "18.52%"
    assert format_percentage(0.0) == "0.00%"
    assert format_percentage(None) == "N/A"


def test_resolve_ticker():
    """Test resolution of Indian stock tickers for NSE and BSE."""
    assert resolve_ticker("TATAMOTORS") == "TATAMOTORS.NS"
    assert resolve_ticker("infy") == "INFY.NS"
    assert resolve_ticker("RELIANCE.NS") == "RELIANCE.NS"
    assert resolve_ticker("500570", exchange="BSE") == "500570.BO"
    assert resolve_ticker("TCS.BO") == "TCS.BO"
