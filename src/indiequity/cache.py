"""
Local caching layer for financial statement data using SQLite.

Prevents rate-limiting and IP bans by public data sources (Yahoo Finance)
when screening across large universes of Indian stocks.
"""

import io
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd


class FinancialCache:
    """
    Lightweight SQLite cache for financial profiles and statement DataFrames.
    Default cache location: ~/.cache/indiequity/market_cache.db
    """

    def __init__(self, db_path: Optional[str] = None, ttl_seconds: int = 86400):
        """
        Args:
            db_path: Path to SQLite database file. Defaults to user cache folder.
            ttl_seconds: Cache validity duration in seconds (defaults to 24 hours).
        """
        if db_path is None:
            cache_dir = Path.home() / ".cache" / "indiequity"
            cache_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = str(cache_dir / "market_cache.db")
        else:
            self.db_path = db_path
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self.ttl_seconds = ttl_seconds
        self._init_db()

    def _init_db(self) -> None:
        """Create tables if they do not exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS company_cache (
                    symbol TEXT PRIMARY KEY,
                    updated_at REAL,
                    info_json TEXT,
                    income_json TEXT,
                    balance_json TEXT,
                    cashflow_json TEXT
                )
            """)
            conn.commit()

    def get(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached company financial statements if not expired.

        Returns:
            Dictionary with parsed DataFrames and info dictionary, or None if expired/missing.
        """
        symbol_clean = symbol.strip().upper()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT updated_at, info_json, income_json, balance_json, cashflow_json FROM company_cache WHERE symbol = ?",
                (symbol_clean,),
            )
            row = cursor.fetchone()

        if not row:
            return None

        updated_at, info_json, inc_json, bal_json, cf_json = row
        # Check TTL
        if (time.time() - updated_at) > self.ttl_seconds:
            return None

        try:
            info = json.loads(info_json) if info_json else {}
            income_stmt = pd.read_json(io.StringIO(inc_json)) if inc_json else pd.DataFrame()
            balance_sheet = pd.read_json(io.StringIO(bal_json)) if bal_json else pd.DataFrame()
            cash_flow = pd.read_json(io.StringIO(cf_json)) if cf_json else pd.DataFrame()
            return {
                "info": info,
                "income_stmt": income_stmt,
                "balance_sheet": balance_sheet,
                "cash_flow": cash_flow,
            }
        except Exception:
            return None

    def set(
        self,
        symbol: str,
        info: Dict[str, Any],
        income_stmt: pd.DataFrame,
        balance_sheet: pd.DataFrame,
        cash_flow: pd.DataFrame,
    ) -> None:
        """Store financial statements in local SQLite cache."""
        symbol_clean = symbol.strip().upper()
        info_json = json.dumps(info, default=str)
        inc_json = income_stmt.to_json() if income_stmt is not None else ""
        bal_json = balance_sheet.to_json() if balance_sheet is not None else ""
        cf_json = cash_flow.to_json() if cash_flow is not None else ""

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO company_cache (
                    symbol, updated_at, info_json, income_json, balance_json, cashflow_json
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (symbol_clean, time.time(), info_json, inc_json, bal_json, cf_json))
            conn.commit()

    def clear(self) -> None:
        """Clear all entries in the cache."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM company_cache")
            conn.commit()
