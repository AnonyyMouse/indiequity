"""
indiequity: Indian Equity Research Toolkit
=========================================
A Python toolkit for small-cap & mid-cap equity research on Indian markets (NSE & BSE).
Provides:
- Public fundamental financial statement ingestion with chronological sorting
- Core ratios: ROE, ROCE, ROIC, Debt/Equity, Interest Coverage, EV/EBITDA, FCF Yield
- Non-cash Operating Working Capital (NWC) and ΔNWC tracking
- 3-Stage and 5-Stage DuPont decomposition
- Configurable stock screening with preset Indian investment strategies
- 5-Year and 10-Year FCFF DCF valuation with sustainable terminal reinvestment & 2D sensitivity
- Bank and Financial Institution Residual Income Model (RIM) & Justified P/B
- Transparent Model Routing (Banks, Lending NBFCs, AMCs, Conglomerates, Industrials)
- Reverse DCF & Scenario Analysis (Bear/Base/Bull)
- Visual charting and multi-tab Excel/CSV export
"""

from indiequity.cache import FinancialCache
from indiequity.charts import plot_dcf_waterfall, plot_dupont_5stage
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
from indiequity.data import (
    BusinessCategory,
    FinancialProfile,
    classify_business,
    fetch_company_data,
)
from indiequity.dcf import (
    AnnualForecast,
    BankAnnualForecast,
    DCFResult,
    EnterpriseToEquityBridge,
    ForecastAssumptions,
    InvalidValuationInputError,
    ResidualIncomeResult,
    UnsupportedValuationModelError,
    calculate_dcf,
    calculate_fcff_item,
    calculate_residual_income_valuation,
    estimate_wacc,
    generate_sensitivity_matrix,
    reverse_dcf,
    run_scenario_analysis,
)
from indiequity.dupont import (
    DuPont3Stage,
    DuPont5Stage,
    decompose_dupont_3stage,
    decompose_dupont_5stage,
)
from indiequity.export import export_screener_to_excel, export_to_csv, export_to_excel
from indiequity.ratios import (
    CapitalStructure,
    RatioMetrics,
    WorkingCapitalMetrics,
    calculate_ratios,
)
from indiequity.screener import (
    NIFTY_50_TICKERS,
    NIFTY_SMALLCAP_PRESET,
    PRESET_DEEP_VALUE,
    PRESET_HIGH_COMPOUNDER,
    PRESET_QUALITY_SMALLCAP,
    ScreenerCriteria,
    StockScreener,
)

__version__ = "0.2.2"
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
    "BusinessCategory",
    "classify_business",
    "FinancialProfile",
    "fetch_company_data",
    "CapitalStructure",
    "WorkingCapitalMetrics",
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
    "AnnualForecast",
    "BankAnnualForecast",
    "EnterpriseToEquityBridge",
    "ForecastAssumptions",
    "DCFResult",
    "ResidualIncomeResult",
    "InvalidValuationInputError",
    "UnsupportedValuationModelError",
    "calculate_fcff_item",
    "calculate_dcf",
    "calculate_residual_income_valuation",
    "reverse_dcf",
    "run_scenario_analysis",
    "estimate_wacc",
    "generate_sensitivity_matrix",
    "plot_dupont_5stage",
    "plot_dcf_waterfall",
    "export_to_excel",
    "export_screener_to_excel",
    "export_to_csv",
]
