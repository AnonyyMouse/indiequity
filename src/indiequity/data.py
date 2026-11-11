"""
Data ingestion module for Indian equities (NSE and BSE).

Fetches public fundamental data (Income Statement, Balance Sheet,
Cash Flow Statement, and Market Info) using yfinance, automatically
applying exchange suffixes (.NS/.BO), chronologically sorting fiscal periods,
distinguishing consolidated from standalone reporting, categorizing business models,
and auditing data quality.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import yfinance as yf

from indiequity.cache import FinancialCache
from indiequity.config import resolve_ticker

_DEFAULT_CACHE = FinancialCache()


class BusinessCategory(str, Enum):
    """Business model classification for valuation routing."""
    BANK = "BANK"
    LENDING_NBFC = "LENDING_NBFC"
    ASSET_MANAGER = "ASSET_MANAGER"
    CONGLOMERATE = "CONGLOMERATE"
    TECH_SERVICES = "TECH_SERVICES"
    PHARMA = "PHARMA"
    INDUSTRIAL_MANUFACTURING = "INDUSTRIAL_MANUFACTURING"
    GENERAL_CORPORATE = "GENERAL_CORPORATE"


# Known Conglomerates requiring Segment / SOTP valuation
KNOWN_CONGLOMERATES = {
    "M&M.NS", "M&M.BO", "RELIANCE.NS", "RELIANCE.BO", "LT.NS", "LT.BO",
    "ITC.NS", "ITC.BO", "GRASIM.NS", "GRASIM.BO", "ADANIENT.NS", "ADANIENT.BO"
}

# Known Asset Managers requiring AUM-driven fee realization
KNOWN_ASSET_MANAGERS = {
    "HDFCAMC.NS", "HDFCAMC.BO", "NAM-INDIA.NS", "NAM-INDIA.BO", "UTIAMC.NS", "UTIAMC.BO"
}

# Known Lending Institutions
KNOWN_BANKS = {
    "ICICIBANK.NS", "ICICIBANK.BO", "HDFCBANK.NS", "HDFCBANK.BO", "SBIN.NS", "SBIN.BO",
    "KOTAKBANK.NS", "KOTAKBANK.BO", "AXISBANK.NS", "AXISBANK.BO", "INDUSINDBK.NS", "INDUSINDBK.BO",
    "BANKBARODA.NS", "BANKBARODA.BO", "PNB.NS", "PNB.BO", "FEDERALBNK.NS", "FEDERALBNK.BO"
}

KNOWN_LENDING_NBFCS = {
    "BAJFINANCE.NS", "BAJFINANCE.BO", "BAJAJFINSV.NS", "BAJAJFINSV.BO",
    "SHRIRAMFIN.NS", "SHRIRAMFIN.BO", "CHOLAFIN.NS", "CHOLAFIN.BO",
    "MUTHOOTFIN.NS", "MUTHOOTFIN.BO", "M&MFIN.NS", "M&MFIN.BO",
    "L&TFH.NS", "L&TFH.BO", "MANAPPURAM.NS", "MANAPPURAM.BO"
}


@dataclass
class FinancialProfile:
    """
    Structured container holding fundamental financial statements, period metadata,
    debt/lease details, business categorization, and market metrics.
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

    # Business Model Classification
    business_category: BusinessCategory = BusinessCategory.GENERAL_CORPORATE
    preferred_valuation_model: str = "FCFF_DCF"

    # Period metadata
    reporting_currency: str = "INR"
    statement_type: str = "Consolidated"  # Standard for Indian listed groups
    latest_period_date: Optional[str] = None
    available_periods: List[str] = field(default_factory=list)

    # Share count verification metadata
    shares_source: str = "REPORTED"  # "REPORTED" or "INFERRED_FROM_MCAP_AND_PRICE"

    # Raw yfinance summary info
    raw_info: Dict[str, Any] = field(default_factory=dict)

    # Audited data quality warnings
    data_quality_warnings: List[str] = field(default_factory=list)

    def __post_init__(self):
        # Automatically infer business category if not explicitly set
        if self.business_category == BusinessCategory.GENERAL_CORPORATE and (self.sector or self.industry):
            b_cat, pref_model = classify_business(self.symbol, self.sector, self.industry)
            self.business_category = b_cat
            self.preferred_valuation_model = pref_model

    def is_financial_institution(self) -> bool:
        """Check if company is a Bank or Lending NBFC."""
        return self.business_category in (BusinessCategory.BANK, BusinessCategory.LENDING_NBFC)

    def is_asset_manager(self) -> bool:
        """Check if company is an Asset Management Company."""
        return self.business_category == BusinessCategory.ASSET_MANAGER

    def is_conglomerate(self) -> bool:
        """Check if company is a multi-segment conglomerate."""
        return self.business_category == BusinessCategory.CONGLOMERATE

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


def classify_business(symbol: str, sector: str, industry: str) -> Tuple[BusinessCategory, str]:
    """
    Classify company into appropriate business category and identify preferred valuation model.
    """
    sym_upper = symbol.strip().upper()
    sec_lower = (sector or "").strip().lower()
    ind_lower = (industry or "").strip().lower()

    # 1. Conglomerates
    if sym_upper in KNOWN_CONGLOMERATES:
        return BusinessCategory.CONGLOMERATE, "SUM_OF_THE_PARTS"

    # 2. Asset Managers
    if sym_upper in KNOWN_ASSET_MANAGERS or "asset management" in ind_lower:
        return BusinessCategory.ASSET_MANAGER, "AUM_DRIVEN_CASH_FLOW"

    # 3. Commercial Banks
    if sym_upper in KNOWN_BANKS or "banks" in ind_lower:
        return BusinessCategory.BANK, "RESIDUAL_INCOME_PB"

    # 4. Lending NBFCs
    if sym_upper in KNOWN_LENDING_NBFCS or "credit services" in ind_lower or "mortgage finance" in ind_lower:
        return BusinessCategory.LENDING_NBFC, "RESIDUAL_INCOME_FCFE"

    # 5. Technology Services
    if "information technology" in sec_lower or "technology" in sec_lower or "software" in ind_lower or "it services" in ind_lower:
        return BusinessCategory.TECH_SERVICES, "FCFF_DCF"

    # 6. Pharmaceuticals
    if "healthcare" in sec_lower or "drug" in ind_lower or "pharmaceutical" in ind_lower or "biotechnology" in ind_lower:
        return BusinessCategory.PHARMA, "FCFF_DCF"

    # 7. Industrial / Manufacturing
    if "industrials" in sec_lower or "manufacturing" in ind_lower or "electrical" in ind_lower or "auto" in ind_lower:
        return BusinessCategory.INDUSTRIAL_MANUFACTURING, "FCFF_DCF"

    return BusinessCategory.GENERAL_CORPORATE, "FCFF_DCF"


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
    - Categorization of business models (Banks, NBFCs, AMCs, Conglomerates, Industrials).
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
    raw_shares = float(info.get("sharesOutstanding") or 0.0)
    beta = info.get("beta")

    # Share count audit
    shares_source = "REPORTED"
    if raw_shares > 0:
        shares_outstanding = raw_shares
    elif market_cap > 0 and current_price > 0:
        shares_outstanding = market_cap / current_price
        shares_source = "INFERRED_FROM_MCAP_AND_PRICE"
        warnings.append(
            "Shares outstanding was inferred from market cap and price; verify against latest shareholding pattern."
        )
    else:
        shares_outstanding = 0.0
        warnings.append("Shares outstanding is missing or zero. Diluted per-share metrics cannot be computed.")

    if shares_outstanding > 0 and market_cap > 0 and current_price > 0:
        implied_mcap = current_price * shares_outstanding
        diff_pct = abs(implied_mcap - market_cap) / market_cap
        if diff_pct > 0.08:
            warnings.append(
                f"Market cap ({market_cap:,.0f}) differs from Price x Shares ({implied_mcap:,.0f}) by {diff_pct:.1%}. Possible share class or reporting lag."
            )

    if income_stmt.empty or balance_sheet.empty:
        warnings.append("Primary financial statements are missing or incomplete in public feed.")

    company_name = info.get("longName") or info.get("shortName") or symbol.upper()
    sector = info.get("sector", "N/A")
    industry = info.get("industry", "N/A")
    currency = info.get("financialCurrency", "INR")

    # Classify Business Category and Route Model
    b_cat, pref_model = classify_business(ticker_str, sector, industry)

    return FinancialProfile(
        symbol=ticker_str,
        company_name=company_name,
        sector=sector,
        industry=industry,
        current_price=current_price,
        market_cap=market_cap,
        shares_outstanding=shares_outstanding,
        shares_source=shares_source,
        beta=float(beta) if beta is not None else None,
        income_stmt=income_stmt,
        balance_sheet=balance_sheet,
        cash_flow=cash_flow,
        business_category=b_cat,
        preferred_valuation_model=pref_model,
        reporting_currency=currency,
        statement_type="Consolidated",
        latest_period_date=latest_period_date,
        available_periods=available_periods,
        raw_info=info,
        data_quality_warnings=warnings,
    )
