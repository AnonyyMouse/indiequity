"""
Data ingestion module for Indian equities (NSE and BSE).

Fetches public fundamental data (Income Statement, Balance Sheet,
Cash Flow Statement, and Market Info) using yfinance, automatically
applying exchange suffixes (.NS/.BO) and normalizing metrics.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional
import pandas as pd
import yfinance as yf

from indiequity.config import resolve_ticker


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
        
        # Look for matching label (case-insensitive substring or exact match)
        for label in row_labels:
            if label in statement.index:
                val = statement.loc[label].iloc[0]
                if pd.notna(val):
                    return float(val)
            # Try case-insensitive lookup
            for idx in statement.index:
                if str(idx).lower() == label.lower():
                    val = statement.loc[idx].iloc[0]
                    if pd.notna(val):
                        return float(val)
        return default


def fetch_company_data(symbol: str, exchange: str = "NSE") -> FinancialProfile:
    """
    Fetch comprehensive fundamental statements and market data for an Indian company.

    Args:
        symbol: Stock ticker symbol (e.g. 'TATAMOTORS', 'INFY', '500570')
        exchange: Exchange code ('NSE' or 'BSE'). Defaults to 'NSE'.

    Returns:
        FinancialProfile object populated with statements and market data.

    Raises:
        ValueError: If symbol data cannot be retrieved from public sources.
    """
    ticker_str = resolve_ticker(symbol, exchange)
    ticker = yf.Ticker(ticker_str)
    
    # Fetch info dictionary
    try:
        info = ticker.info or {}
    except Exception:
        info = {}

    # Extract price and shares
    current_price = info.get("currentPrice") or info.get("regularMarketPrice") or 0.0
    market_cap = info.get("marketCap") or 0.0
    shares_outstanding = info.get("sharesOutstanding") or 0.0
    company_name = info.get("longName") or info.get("shortName") or symbol.upper()
    sector = info.get("sector", "N/A")
    industry = info.get("industry", "N/A")
    beta = info.get("beta")

    # Fetch financial statements (annual)
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
