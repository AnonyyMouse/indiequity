"""
Financial ratio calculations for Indian equity research.

Provides standardized implementations for:
- Return on Equity (ROE)
- Return on Capital Employed (ROCE)
- Debt-to-Equity (D/E)
- Interest Coverage Ratio
- Free Cash Flow (FCF) and FCF Yield
- Enterprise Value to EBITDA (EV/EBITDA)
"""

from dataclasses import dataclass
from typing import Optional
from indiequity.data import FinancialProfile


@dataclass
class RatioMetrics:
    """Container holding computed financial and valuation ratios."""
    symbol: str
    company_name: str
    revenue: float
    ebit: float
    ebitda: float
    net_income: float
    total_assets: float
    current_liabilities: float
    total_equity: float
    total_debt: float
    cash_and_equivalents: float
    operating_cash_flow: float
    capex: float
    free_cash_flow: float
    market_cap: float
    enterprise_value: float

    # Core Computed Ratios
    roe: Optional[float]
    roce: Optional[float]
    debt_to_equity: Optional[float]
    interest_coverage: Optional[float]
    fcf_yield: Optional[float]
    ev_to_ebitda: Optional[float]
    net_profit_margin: Optional[float]
    operating_margin: Optional[float]
    asset_turnover: Optional[float]


def calculate_ratios(profile: FinancialProfile) -> RatioMetrics:
    """
    Compute comprehensive fundamental financial ratios from a FinancialProfile.

    Formulas and References:
    1. ROE = Net Income / Total Equity
    2. ROCE = EBIT / (Total Assets - Current Liabilities)
       Reference: Damodaran on Valuation; NISM Series XV Equity Research.
    3. Debt to Equity = Total Debt / Total Equity
    4. Interest Coverage = EBIT / Interest Expense
       Reference: Graham & Dodd, Security Analysis.
    5. FCF = Cash Flow from Operations - CapEx
    6. FCF Yield = FCF / Market Cap
    7. EV / EBITDA = (Market Cap + Total Debt - Cash) / EBITDA

    Args:
        profile: Populated FinancialProfile containing financial statements.

    Returns:
        RatioMetrics with all calculated values and ratios.
    """
    bs = profile.balance_sheet
    inc = profile.income_stmt
    cf = profile.cash_flow

    # Extract Income Statement items
    revenue = profile.get_latest(inc, ["Total Revenue", "Operating Revenue", "Revenue"])
    ebit = profile.get_latest(inc, ["EBIT", "Operating Income", "Operating Profit"])
    ebitda = profile.get_latest(inc, ["EBITDA", "Normalized EBITDA"])
    if ebitda == 0.0 and ebit != 0.0:
        # Fallback: estimate EBITDA = EBIT + Depreciation & Amortization
        dna = profile.get_latest(cf, [
            "Depreciation And Amortization",
            "Depreciation & Amortization",
            "Depreciation Amortization Depletion",
        ])
        ebitda = ebit + abs(dna)

    net_income = profile.get_latest(inc, [
        "Net Income Common Stockholders",
        "Net Income",
        "Net Income Continuous Operations",
    ])
    interest_expense = abs(profile.get_latest(inc, [
        "Interest Expense",
        "Interest Expense Non Operating",
        "Finance Costs",
    ]))

    # Extract Balance Sheet items
    total_assets = profile.get_latest(bs, ["Total Assets"])
    current_liabilities = profile.get_latest(bs, ["Current Liabilities", "Total Current Liabilities"])
    total_equity = profile.get_latest(bs, [
        "Stockholders Equity",
        "Total Stockholder Equity",
        "Common Stock Equity",
        "Total Equity",
    ])
    total_debt = profile.get_latest(bs, [
        "Total Debt",
        "Long Term Debt And Capital Lease Obligation",
    ])
    if total_debt == 0.0:
        # Add Long Term Debt + Short Term Debt
        lt_debt = profile.get_latest(bs, ["Long Term Debt", "Long Term Debt Non Current"])
        st_debt = profile.get_latest(bs, ["Current Debt", "Short Long Term Debt", "Current Debt And Capital Lease Obligation"])
        total_debt = lt_debt + st_debt

    cash_and_equivalents = profile.get_latest(bs, [
        "Cash And Cash Equivalents",
        "Cash Cash Equivalents And Short Term Investments",
        "Cash",
    ])

    # Extract Cash Flow items
    operating_cash_flow = profile.get_latest(cf, [
        "Operating Cash Flow",
        "Cash Flow From Continuing Operating Activities",
    ])
    capex = abs(profile.get_latest(cf, [
        "Capital Expenditure",
        "Capital Expenditures",
        "Purchase Of PPE",
    ]))
    free_cash_flow = operating_cash_flow - capex

    # Market Valuation items
    market_cap = profile.market_cap
    if market_cap == 0.0 and profile.current_price > 0 and profile.shares_outstanding > 0:
        market_cap = profile.current_price * profile.shares_outstanding

    enterprise_value = profile.raw_info.get("enterpriseValue") or (market_cap + total_debt - cash_and_equivalents)

    # 1. Return on Equity (ROE)
    roe = (net_income / total_equity) if total_equity > 0 else None

    # 2. Return on Capital Employed (ROCE)
    # Capital Employed = Total Assets - Current Liabilities
    capital_employed = total_assets - current_liabilities
    roce = (ebit / capital_employed) if capital_employed > 0 else None

    # 3. Debt to Equity (D/E)
    debt_to_equity = (total_debt / total_equity) if total_equity > 0 else None

    # 4. Interest Coverage Ratio
    interest_coverage = (ebit / interest_expense) if interest_expense > 0 else (999.0 if ebit > 0 else None)

    # 5. Free Cash Flow Yield
    fcf_yield = (free_cash_flow / market_cap) if market_cap > 0 else None

    # 6. EV / EBITDA
    ev_to_ebitda = (enterprise_value / ebitda) if (ebitda > 0 and enterprise_value > 0) else None

    # Margins and Turnover
    net_profit_margin = (net_income / revenue) if revenue > 0 else None
    operating_margin = (ebit / revenue) if revenue > 0 else None
    asset_turnover = (revenue / total_assets) if total_assets > 0 else None

    return RatioMetrics(
        symbol=profile.symbol,
        company_name=profile.company_name,
        revenue=revenue,
        ebit=ebit,
        ebitda=ebitda,
        net_income=net_income,
        total_assets=total_assets,
        current_liabilities=current_liabilities,
        total_equity=total_equity,
        total_debt=total_debt,
        cash_and_equivalents=cash_and_equivalents,
        operating_cash_flow=operating_cash_flow,
        capex=capex,
        free_cash_flow=free_cash_flow,
        market_cap=market_cap,
        enterprise_value=enterprise_value,
        roe=roe,
        roce=roce,
        debt_to_equity=debt_to_equity,
        interest_coverage=interest_coverage,
        fcf_yield=fcf_yield,
        ev_to_ebitda=ev_to_ebitda,
        net_profit_margin=net_profit_margin,
        operating_margin=operating_margin,
        asset_turnover=asset_turnover,
    )
