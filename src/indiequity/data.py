"""
Data ingestion module for Indian equities (NSE and BSE).

Fetches public fundamental data (Income Statement, Balance Sheet,
Cash Flow Statement, and Market Info) using yfinance, automatically
applying exchange suffixes (.NS/.BO), checking local cache, and
normalizing metrics.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional
import pandas as pd
import yfinance as yf

from indiequity.cache import FinancialCache
from indiequity.config import resolve_ticker

# Shared default cache instance
_DEFAULT_CACHE = FinancialCache()


@dataclass
class FinancialProfile:
    """
    Structured container holding fundamental financial statements and market metrics.
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

    # Financial statements (pandas DataFrames: items as rows, fiscal dates as columns)
    income_stmt: pd.DataFrame
    balance_sheet: pd.DataFrame
    cash_flow: pd.DataFrame

    # Raw yfinance summary info dictionary
    raw_info: Dict[str, Any]

    def get_latest(self, statement: pd.DataFrame, row_labels: list[str], default: float = 0.0) -> float:
        """
        Safely fetch the most recent annual value for a given line item,
        checking multiple standard alias names across accounting standards.
        """
        if statement is None or statement.empty:
            return default

        for label in row_labels:
            if label in statement.index:
                val = statement.loc[label].iloc[0]
                if pd.notna(val):
                    return float(val)
            for idx in statement.index:
                if str(idx).lower() == label.lower():
                    val = statement.loc[idx].iloc[0]
                    if pd.notna(val):
                        return float(val)
        return default


def fetch_company_data(
    symbol: str,
    exchange: str = "NSE",
    use_cache: bool = True,
    cache: Optional[FinancialCache] = None,
) -> FinancialProfile:
    """
    Fetch comprehensive fundamental statements and market data for an Indian company.

    Args:
        symbol: Stock ticker symbol (e.g. 'TATAMOTORS', 'INFY', '500570')
        exchange: Exchange code ('NSE' or 'BSE'). Defaults to 'NSE'.
        use_cache: If True, retrieves and stores data in local SQLite cache.
        cache: Optional custom FinancialCache instance.

    Returns:
        FinancialProfile object populated with statements and market data.
    """
    ticker_str = resolve_ticker(symbol, exchange)
    active_cache = cache or _DEFAULT_CACHE

    # 1. Check local cache first
    if use_cache:
        cached = active_cache.get(ticker_str)
        if cached is not None:
            info = cached.get("info", {})
            current_price = info.get("currentPrice") or info.get("regularMarketPrice") or 0.0
            market_cap = info.get("marketCap") or 0.0
            shares_outstanding = info.get("sharesOutstanding") or 0.0
            company_name = info.get("longName") or info.get("shortName") or symbol.upper()
            sector = info.get("sector", "N/A")
            industry = info.get("industry", "N/A")
            beta = info.get("beta")

            return FinancialProfile(
                symbol=ticker_str,
                company_name=company_name,
                sector=sector,
                industry=industry,
                current_price=float(current_price),
                market_cap=float(market_cap),
                shares_outstanding=float(shares_outstanding),
                beta=float(beta) if beta is not None else None,
                income_stmt=cached.get("income_stmt", pd.DataFrame()),
                balance_sheet=cached.get("balance_sheet", pd.DataFrame()),
                cash_flow=cached.get("cash_flow", pd.DataFrame()),
                raw_info=info,
            )

    # 2. Fetch from network
    ticker = yf.Ticker(ticker_str)
    try:
        info = ticker.info or {}
    except Exception:
        info = {}

    current_price = info.get("currentPrice") or info.get("regularMarketPrice") or 0.0
    market_cap = info.get("marketCap") or 0.0
    shares_outstanding = info.get("sharesOutstanding") or 0.0
    company_name = info.get("longName") or info.get("shortName") or symbol.upper()
    sector = info.get("sector", "N/A")
    industry = info.get("industry", "N/A")
    beta = info.get("beta")

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

    # 3. Save to cache
    if use_cache:
        active_cache.set(
            ticker_str,
            info=info,
            income_stmt=income_stmt,
            balance_sheet=balance_sheet,
            cash_flow=cash_flow,
        )

    return FinancialProfile(
        symbol=ticker_str,
        company_name=company_name,
        sector=sector,
        industry=industry,
        current_price=float(current_price),
        market_cap=float(market_cap),
        shares_outstanding=float(shares_outstanding),
        beta=float(beta) if beta is not None else None,
        income_stmt=income_stmt,
        balance_sheet=balance_sheet,
        cash_flow=cash_flow,
        raw_info=info,
    )
