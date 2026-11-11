"""
indiequity: Indian Equity Research Toolkit
=========================================
A Python toolkit for small-cap & mid-cap equity research on Indian markets (NSE & BSE).
Provides:
- Public fundamental financial statement ingestion
- Core ratios: ROE, ROCE, Debt/Equity, Interest Coverage, EV/EBITDA, FCF Yield
- 3-Stage and 5-Stage DuPont decomposition
- Configurable stock screening with preset Indian investment strategies
- 5-Year FCFF DCF valuation engine with 2D sensitivity analysis
- Visual charting and Excel/CSV export
"""

from indiequity.cache import FinancialCache
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
from indiequity.data import FinancialProfile, fetch_company_data
from indiequity.dupont import (
    DuPont3Stage,
    DuPont5Stage,
    decompose_dupont_3stage,
    decompose_dupont_5stage,
)
from indiequity.ratios import RatioMetrics, calculate_ratios
from indiequity.screener import (
    NIFTY_50_TICKERS,
    NIFTY_SMALLCAP_PRESET,
    PRESET_DEEP_VALUE,
    PRESET_HIGH_COMPOUNDER,
    PRESET_QUALITY_SMALLCAP,
    ScreenerCriteria,
    StockScreener,
)

__version__ = "0.1.0"
__author__ = "Ayush"

__all__ = [
    "__version__",
    "DEFAULT_RISK_FREE_RATE",
    "DEFAULT_EQUITY_RISK_PREMIUM",
    "DEFAULT_TERMINAL_GROWTH_RATE",
    "DEFAULT_TAX_RATE",
    "CRORE_DIVISOR",
    "format_crores",
    "format_percentage",
    "resolve_ticker",
    "FinancialCache",
    "FinancialProfile",
    "fetch_company_data",
    "RatioMetrics",
    "calculate_ratios",
    "DuPont3Stage",
    "DuPont5Stage",
    "decompose_dupont_3stage",
    "decompose_dupont_5stage",
    "NIFTY_50_TICKERS",
    "NIFTY_SMALLCAP_PRESET",
    "ScreenerCriteria",
    "StockScreener",
    "PRESET_QUALITY_SMALLCAP",
    "PRESET_DEEP_VALUE",
    "PRESET_HIGH_COMPOUNDER",
]
