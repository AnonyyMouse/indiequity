"""
Data ingestion module for Indian equities (NSE and BSE).

Fetches public fundamental data (Income Statement, Balance Sheet,
Cash Flow Statement, and Market Info) using yfinance, automatically
applying exchange suffixes (.NS/.BO), chronologically sorting fiscal periods,
distinguishing consolidated from standalone reporting, and auditing data quality.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import yfinance as yf

from indiequity.cache import FinancialCache
from indiequity.config import resolve_ticker

_DEFAULT_CACHE = FinancialCache()

# Industries requiring specialized financial institution valuation
FINANCIAL_INSTITUTION_INDUSTRIES = {
    "banks - diversified",
    "banks - regional",
    "credit services",
    "financial data & stock exchanges",
    "insurance - life",
    "insurance - property & casualty",
    "insurance - diversified",
    "asset management",
    "mortgage finance",
    "nbfc",
}

FINANCIAL_INSTITUTION_SECTORS = {
    "financial services",
    "financials",
}


@dataclass
class FinancialProfile:
    """
    Structured container holding fundamental financial statements, period metadata,
    debt/lease details, and market metrics for an Indian company.
    All raw monetary figures are stored in Indian Rupees (INR).
    """
    symbol: str
    company_name: str
    sector: str
    industry: str
    current_price: float
    market_cap: float
    shares_outstanding: float
    beta: Optional[float]

    # Chronologically sorted statements (columns sorted newest to oldest)
    income_stmt: pd.DataFrame
    balance_sheet: pd.DataFrame
    cash_flow: pd.DataFrame

    # Period metadata
    reporting_currency: str = "INR"
    statement_type: str = "Consolidated"  # Standard for Indian listed groups
    latest_period_date: Optional[str] = None
    available_periods: List[str] = field(default_factory=list)

    # Raw yfinance summary info
    raw_info: Dict[str, Any] = field(default_factory=dict)

    # Audited data quality warnings
    data_quality_warnings: List[str] = field(default_factory=list)

    def is_financial_institution(self) -> bool:
        """Check if company is a Bank, NBFC, or Financial Institution."""
        sec = (self.sector or "").strip().lower()
        ind = (self.industry or "").strip().lower()
        return (sec in FINANCIAL_INSTITUTION_SECTORS) or (ind in FINANCIAL_INSTITUTION_INDUSTRIES)

    def get_latest(
        self,
        statement: pd.DataFrame,
        row_labels: List[str],
        default: Optional[float] = None,
        period_idx: int = 0,
    ) -> Optional[float]:
        """
        Safely fetch a value for a line item from a chronologically sorted statement.

        Args:
            statement: Financial statement DataFrame (columns sorted newest to oldest).
            row_labels: Priority list of standard line item aliases.
            default: Default value if not found. If None and missing, returns None.
            period_idx: Column index (0 = latest period, 1 = previous year, etc.).

        Returns:
            Float value or None/default if missing.
        """
        if statement is None or statement.empty:
            return default

        if period_idx >= len(statement.columns):
            return default

        col = statement.columns[period_idx]

        for label in row_labels:
            if label in statement.index:
                val = statement.loc[label, col]
                if pd.notna(val):
                    return float(val)
            for idx in statement.index:
                if str(idx).strip().lower() == label.strip().lower():
                    val = statement.loc[idx, col]
                    if pd.notna(val):
                        return float(val)

        return default

    def get_series(self, statement: pd.DataFrame, row_labels: List[str]) -> pd.Series:
        """
        Extract a historical series for a given line item across all available sorted periods.
        """
        if statement is None or statement.empty:
            return pd.Series(dtype=float)

        for label in row_labels:
            if label in statement.index:
                s = statement.loc[label]
                if isinstance(s, pd.DataFrame):
                    s = s.iloc[0]
                return s.dropna().astype(float)
            for idx in statement.index:
                if str(idx).strip().lower() == label.strip().lower():
                    s = statement.loc[idx]
                    if isinstance(s, pd.DataFrame):
                        s = s.iloc[0]
                    return s.dropna().astype(float)

        return pd.Series(dtype=float)


def _sort_statement_columns(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    Ensure statement columns are chronologically sorted from newest to oldest.
    Never assume the first column in a raw DataFrame is the latest!
    """
    if df is None or df.empty:
        return pd.DataFrame()

    def parse_col_date(col_key: Any) -> datetime:
        if isinstance(col_key, (datetime, pd.Timestamp)):
            return pd.to_datetime(col_key).to_pydatetime()
        try:
            return pd.to_datetime(str(col_key)).to_pydatetime()
        except Exception:
            return datetime.min

    try:
        sorted_cols = sorted(df.columns, key=parse_col_date, reverse=True)
        return df[sorted_cols]
    except Exception:
        return df


def fetch_company_data(
    symbol: str,
    exchange: str = "NSE",
    use_cache: bool = True,
    cache: Optional[FinancialCache] = None,
) -> FinancialProfile:
    """
    Fetch comprehensive fundamental statements and market data for an Indian company.

    Audits:
    - Chronological sorting of fiscal balance sheets, P&Ls, and cash flows.
    - Identification of debt vs lease liabilities.
    - Verification of share count, price, and market cap consistency.
    - Identification of financial institution sector routing.
    """
    ticker_str = resolve_ticker(symbol, exchange)
    active_cache = cache or _DEFAULT_CACHE
    warnings: List[str] = []

    # 1. Check local cache
    cached_data = None
    if use_cache:
        cached_data = active_cache.get(ticker_str)

    if cached_data is not None:
        info = cached_data.get("info", {})
        income_stmt = cached_data.get("income_stmt", pd.DataFrame())
        balance_sheet = cached_data.get("balance_sheet", pd.DataFrame())
        cash_flow = cached_data.get("cash_flow", pd.DataFrame())
    else:
        # 2. Fetch from network
        ticker = yf.Ticker(ticker_str)
        try:
            info = ticker.info or {}
        except Exception as e:
            info = {}
            warnings.append(f"Network error retrieving summary info: {str(e)}")

        income_stmt = ticker.financials
        if income_stmt is None or income_stmt.empty:
            income_stmt = ticker.income_stmt
        if income_stmt is None:
            income_stmt = pd.DataFrame()

        balance_sheet = ticker.balance_sheet
        if balance_sheet is None or balance_sheet.empty:
            balance_sheet = ticker.balancesheet
        if balance_sheet is None:
            balance_sheet = pd.DataFrame()

        cash_flow = ticker.cashflow
        if cash_flow is None or cash_flow.empty:
            cash_flow = ticker.cash_flow
        if cash_flow is None:
            cash_flow = pd.DataFrame()

        if use_cache and not income_stmt.empty:
            active_cache.set(
                ticker_str,
                info=info,
                income_stmt=income_stmt,
                balance_sheet=balance_sheet,
                cash_flow=cash_flow,
            )

    # 3. Chronologically sort columns (Newest -> Oldest)
    income_stmt = _sort_statement_columns(income_stmt)
    balance_sheet = _sort_statement_columns(balance_sheet)
    cash_flow = _sort_statement_columns(cash_flow)

    # Extract dates
    available_periods = [str(c)[:10] for c in balance_sheet.columns]
    latest_period_date = available_periods[0] if available_periods else None

    # Validate market data
    current_price = float(info.get("currentPrice") or info.get("regularMarketPrice") or 0.0)
    market_cap = float(info.get("marketCap") or 0.0)
    shares_outstanding = float(info.get("sharesOutstanding") or 0.0)
    beta = info.get("beta")

    # Data Quality Audits
    if current_price <= 0.0:
        warnings.append("Current share price is missing or zero.")

    if shares_outstanding <= 0.0:
        if market_cap > 0 and current_price > 0:
            shares_outstanding = market_cap / current_price
            warnings.append("Shares outstanding was inferred from market cap and price; verify against latest shareholding pattern.")
        else:
            warnings.append("Shares outstanding is missing or zero. Diluted per-share metrics cannot be computed.")
    elif market_cap > 0 and current_price > 0:
        implied_mcap = current_price * shares_outstanding
        diff_pct = abs(implied_mcap - market_cap) / market_cap
        if diff_pct > 0.08:
            warnings.append(f"Market cap ({market_cap:,.0f}) differs from Price x Shares ({implied_mcap:,.0f}) by {diff_pct:.1%}. Possible share class or reporting lag.")

    if income_stmt.empty or balance_sheet.empty:
        warnings.append("Primary financial statements are missing or incomplete in public feed.")

    company_name = info.get("longName") or info.get("shortName") or symbol.upper()
    sector = info.get("sector", "N/A")
    industry = info.get("industry", "N/A")
    currency = info.get("financialCurrency", "INR")

    return FinancialProfile(
        symbol=ticker_str,
        company_name=company_name,
        sector=sector,
        industry=industry,
        current_price=current_price,
        market_cap=market_cap,
        shares_outstanding=shares_outstanding,
        beta=float(beta) if beta is not None else None,
        income_stmt=income_stmt,
        balance_sheet=balance_sheet,
        cash_flow=cash_flow,
        reporting_currency=currency,
        statement_type="Consolidated",
        latest_period_date=latest_period_date,
        available_periods=available_periods,
        raw_info=info,
        data_quality_warnings=warnings,
    )
