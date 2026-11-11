"""
Unit tests for StockScreener and FinancialCache modules.
"""

import tempfile
import pandas as pd
import pytest
from indiequity.cache import FinancialCache
from indiequity.data import FinancialProfile
from indiequity.ratios import calculate_ratios
from indiequity.screener import (
    PRESET_QUALITY_SMALLCAP,
    ScreenerCriteria,
    StockScreener,
)


def test_financial_cache():
    """Verify SQLite caching stores, retrieves, and expires statement data."""
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        cache = FinancialCache(db_path=tmp.name, ttl_seconds=3600)

        # Initially empty
        assert cache.get("TEST.NS") is None

        # Store test data
        info = {"longName": "Test Co", "currentPrice": 150.0}
        df_inc = pd.DataFrame({"2024": [100.0]}, index=["Revenue"])
        df_bal = pd.DataFrame({"2024": [200.0]}, index=["Total Assets"])
        df_cf = pd.DataFrame({"2024": [30.0]}, index=["Operating Cash Flow"])

        cache.set("TEST.NS", info, df_inc, df_bal, df_cf)

        # Retrieve and verify
        retrieved = cache.get("TEST.NS")
        assert retrieved is not None
        assert retrieved["info"]["longName"] == "Test Co"
        assert retrieved["income_stmt"].loc["Revenue"].iloc[0] == 100.0

        # Clear cache
        cache.clear()
        assert cache.get("TEST.NS") is None


def test_screener_evaluation_passes(mock_indian_smallcap: FinancialProfile):
    """Verify high-quality small-cap passes default quality criteria."""
    metrics = calculate_ratios(mock_indian_smallcap)
    screener = StockScreener(criteria=PRESET_QUALITY_SMALLCAP)

    passed, failures = screener.evaluate(metrics)
    assert passed is True
    assert len(failures) == 0


def test_screener_evaluation_fails_strict_criteria(mock_indian_smallcap: FinancialProfile):
    """Verify company fails when criteria are intentionally made stricter than its metrics."""
    metrics = calculate_ratios(mock_indian_smallcap)

    # Bharat Precision has ROCE = 30%, D/E = 0.33, Mcap = ₹1500 Cr
    strict_criteria = ScreenerCriteria(
        min_roce=0.35,              # Company is 30% -> Should FAIL
        max_debt_to_equity=0.20,    # Company is 0.33 -> Should FAIL
        max_market_cap_cr=1000.0,   # Company is 1500 Cr -> Should FAIL
    )
    screener = StockScreener(criteria=strict_criteria)
    passed, failures = screener.evaluate(metrics)

    assert passed is False
    assert len(failures) == 3
    assert any("ROCE" in f for f in failures)
    assert any("D/E" in f for f in failures)
    assert any("Mcap" in f for f in failures)
