"""
Financial ratio and accounting balance calculations for Indian equity research.

Provides standardized, audited implementations for:
- Operating Working Capital (NWC) excluding cash and debt balances
- Change in Operating Working Capital (ΔNWC)
- Return on Invested Capital (ROIC)
- Return on Capital Employed (ROCE)
- Return on Equity (ROE)
- Debt-to-Equity (D/E) and Lease Obligation tracking
- Interest Coverage Ratio
- Enterprise Value to EBITDA (EV/EBITDA)
"""

from dataclasses import dataclass, field
from typing import List, Optional
from indiequity.data import FinancialProfile


@dataclass
class CapitalStructure:
    """Detailed breakdown of interest-bearing debt, leases, cash, and equity claims."""
    short_term_debt: float
    long_term_debt: float
    total_interest_bearing_debt: float
    lease_liabilities: float
    total_debt_and_leases: float
    cash_and_equivalents: float
    short_term_investments: float
    total_cash_and_liquid_assets: float
    net_debt: float
    minority_interest: float
    preferred_stock: float
    total_stockholders_equity: float
    market_cap: float
    enterprise_value: float


@dataclass
class WorkingCapitalMetrics:
    """Operating working capital items excluding financing and cash balances."""
    current_assets: float
    cash_and_equivalents: float
    short_term_investments: float
    operating_current_assets: float
    current_liabilities: float
    short_term_debt: float
    current_lease_liabilities: float
    operating_current_liabilities: float
    operating_nwc: float
    prev_operating_nwc: Optional[float]
    change_in_operating_nwc: float


@dataclass
class RatioMetrics:
    """Container holding computed financial, capital structure, and valuation ratios."""
    symbol: str
    company_name: str
    period_date: Optional[str]

    # Statement Items
    revenue: float
    ebit: float
    ebitda: float
    net_income: float
    tax_expense: float
    effective_tax_rate: float
    nopat: float
    depreciation_and_amortization: float
    capex: float
    free_cash_flow_reported: float

    # Capital Structure & Working Capital
    capital_structure: CapitalStructure
    working_capital: WorkingCapitalMetrics

    # Computed Return & Solvency Ratios
    roe: Optional[float]
    roce: Optional[float]
    roic: Optional[float]
    debt_to_equity: Optional[float]
    total_debt_to_equity: Optional[float]
    interest_coverage: Optional[float]
    fcf_yield: Optional[float]
    ev_to_ebitda: Optional[float]
    net_profit_margin: Optional[float]
    operating_margin: Optional[float]
    asset_turnover: Optional[float]

    # Accounting Warnings
    warnings: List[str] = field(default_factory=list)

    # Convenience backward-compatible properties
    @property
    def total_assets(self) -> float:
        return self.working_capital.current_assets  # used as fallback in legacy callers

    @property
    def total_debt(self) -> float:
        return self.capital_structure.total_interest_bearing_debt

    @property
    def current_liabilities(self) -> float:
        return self.working_capital.current_liabilities

    @property
    def total_equity(self) -> float:
        return self.capital_structure.total_stockholders_equity

    @property
    def cash_and_equivalents(self) -> float:
        return self.capital_structure.cash_and_equivalents

    @property
    def market_cap(self) -> float:
        return self.capital_structure.market_cap

    @property
    def enterprise_value(self) -> float:
        return self.capital_structure.enterprise_value

    @property
    def operating_cash_flow(self) -> float:
        return self.free_cash_flow_reported + self.capex

    @property
    def free_cash_flow(self) -> float:
        return self.free_cash_flow_reported


def calculate_ratios(profile: FinancialProfile) -> RatioMetrics:
    """
    Compute comprehensive fundamental financial ratios, capital structure details,
    and non-cash operating working capital from a FinancialProfile.
    """
    bs = profile.balance_sheet
    inc = profile.income_stmt
    cf = profile.cash_flow
    warnings = list(profile.data_quality_warnings)

    period_date = profile.latest_period_date

    # 1. Income Statement Line Items (Period 0 = Latest)
    revenue = profile.get_latest(inc, ["Total Revenue", "Operating Revenue", "Revenue"], default=0.0)
    ebit = profile.get_latest(inc, ["EBIT", "Operating Income", "Operating Profit"], default=0.0)
    ebitda = profile.get_latest(inc, ["EBITDA", "Normalized EBITDA"], default=0.0)
    net_income = profile.get_latest(inc, ["Net Income Common Stockholders", "Net Income"], default=0.0)
    tax_expense = profile.get_latest(inc, ["Tax Provision", "Income Tax Expense", "Tax"], default=0.0)
    ebt = profile.get_latest(inc, ["Pretax Income", "Income Before Tax", "Pre Tax Income"], default=0.0)
    interest_expense = abs(profile.get_latest(inc, ["Interest Expense", "Interest Expense Non Operating", "Finance Costs"], default=0.0))

    # Effective tax rate
    if ebt > 0 and tax_expense > 0:
        effective_tax_rate = min(max(tax_expense / ebt, 0.15), 0.35)
    else:
        effective_tax_rate = 0.2517  # Section 115BAA benchmark

    nopat = ebit * (1.0 - effective_tax_rate)

    # 2. Cash Flow Items
    dna = abs(profile.get_latest(cf, [
        "Depreciation And Amortization",
        "Depreciation & Amortization",
        "Depreciation Amortization Depletion",
    ], default=0.0))

    if ebitda == 0.0 and ebit != 0.0:
        ebitda = ebit + dna

    cfo = profile.get_latest(cf, [
        "Operating Cash Flow",
        "Cash Flow From Continuing Operating Activities",
    ], default=0.0)

    capex = abs(profile.get_latest(cf, [
        "Capital Expenditure",
        "Capital Expenditures",
        "Purchase Of PPE",
    ], default=0.0))

    fcf_reported = cfo - capex

    # 3. Balance Sheet Items & Capital Structure (Period 0)
    total_assets = profile.get_latest(bs, ["Total Assets"], default=0.0)
    current_assets = profile.get_latest(bs, ["Current Assets", "Total Current Assets"], default=0.0)
    current_liabilities = profile.get_latest(bs, ["Current Liabilities", "Total Current Liabilities"], default=0.0)
    total_equity = profile.get_latest(bs, [
        "Stockholders Equity",
        "Total Stockholder Equity",
        "Common Stock Equity",
        "Total Equity",
    ], default=0.0)

    # Interest-Bearing Debt breakdown
    st_debt = profile.get_latest(bs, [
        "Current Debt",
        "Current Debt And Capital Lease Obligation",
        "Short Term Debt",
        "Commercial Paper",
    ], default=0.0)

    lt_debt = profile.get_latest(bs, [
        "Long Term Debt",
        "Long Term Debt Non Current",
        "Long Term Debt And Capital Lease Obligation",
    ], default=0.0)

    total_debt = profile.get_latest(bs, ["Total Debt"], default=0.0)
    if total_debt == 0.0:
        total_debt = st_debt + lt_debt

    # Lease liabilities
    current_leases = profile.get_latest(bs, ["Current Capital Lease Obligation", "Current Lease Liabilities"], default=0.0)
    lt_leases = profile.get_latest(bs, ["Long Term Capital Lease Obligation", "Non Current Lease Liabilities", "Capital Lease Obligations"], default=0.0)
    total_leases = current_leases + lt_leases

    # Cash and short-term liquid investments
    cash = profile.get_latest(bs, [
        "Cash And Cash Equivalents",
        "Cash Cash Equivalents And Short Term Investments",
        "Cash",
    ], default=0.0)

    short_term_inv = profile.get_latest(bs, [
        "Other Short Term Investments",
        "Short Term Investments",
    ], default=0.0)
    total_cash_liquid = cash + short_term_inv

    # Other claims
    minority_interest = profile.get_latest(bs, ["Minority Interest", "Non Controlling Interest"], default=0.0)
    preferred_stock = profile.get_latest(bs, ["Preferred Stock", "Preferred Equity"], default=0.0)

    market_cap = profile.market_cap
    if market_cap <= 0 and profile.current_price > 0 and profile.shares_outstanding > 0:
        market_cap = profile.current_price * profile.shares_outstanding

    net_debt = total_debt + total_leases - total_cash_liquid
    enterprise_value = profile.raw_info.get("enterpriseValue") or (market_cap + net_debt + minority_interest + preferred_stock)

    cap_struct = CapitalStructure(
        short_term_debt=st_debt,
        long_term_debt=lt_debt,
        total_interest_bearing_debt=total_debt,
        lease_liabilities=total_leases,
        total_debt_and_leases=total_debt + total_leases,
        cash_and_equivalents=cash,
        short_term_investments=short_term_inv,
        total_cash_and_liquid_assets=total_cash_liquid,
        net_debt=net_debt,
        minority_interest=minority_interest,
        preferred_stock=preferred_stock,
        total_stockholders_equity=total_equity,
        market_cap=market_cap,
        enterprise_value=float(enterprise_value),
    )

    # 4. Operating Working Capital (Excluding Cash & Financing Items)
    # Operating Current Assets = Current Assets - Cash - Short Term Investments
    op_ca = max(0.0, current_assets - total_cash_liquid)
    # Operating Current Liabilities = Current Liabilities - Short Term Debt - Current Leases
    op_cl = max(0.0, current_liabilities - st_debt - current_leases)
    operating_nwc = op_ca - op_cl

    # Prior Period NWC (Period 1 if available)
    prev_nwc = None
    change_in_nwc = 0.0
    if len(bs.columns) > 1:
        prev_ca = profile.get_latest(bs, ["Current Assets", "Total Current Assets"], default=0.0, period_idx=1)
        prev_cash = profile.get_latest(bs, ["Cash And Cash Equivalents", "Cash"], default=0.0, period_idx=1)
        prev_inv = profile.get_latest(bs, ["Other Short Term Investments", "Short Term Investments"], default=0.0, period_idx=1)
        prev_cl = profile.get_latest(bs, ["Current Liabilities", "Total Current Liabilities"], default=0.0, period_idx=1)
        prev_st_debt = profile.get_latest(bs, ["Current Debt", "Short Term Debt"], default=0.0, period_idx=1)
        prev_leases = profile.get_latest(bs, ["Current Capital Lease Obligation", "Current Lease Liabilities"], default=0.0, period_idx=1)

        prev_op_ca = max(0.0, prev_ca - (prev_cash + prev_inv))
        prev_op_cl = max(0.0, prev_cl - (prev_st_debt + prev_leases))
        prev_nwc = prev_op_ca - prev_op_cl
        change_in_nwc = operating_nwc - prev_nwc
    else:
        # Fallback from Cash Flow statement if 2-year balance sheet not in feed
        cf_wc_change = profile.get_latest(cf, ["Change In Working Capital", "Changes In Working Capital"], default=None)
        if cf_wc_change is not None:
            # An outflow in CF represents an increase in NWC
            change_in_nwc = -float(cf_wc_change)
            warnings.append("Change in operating working capital was sourced from the cash flow statement.")

    working_cap = WorkingCapitalMetrics(
        current_assets=current_assets,
        cash_and_equivalents=cash,
        short_term_investments=short_term_inv,
        operating_current_assets=op_ca,
        current_liabilities=current_liabilities,
        short_term_debt=st_debt,
        current_lease_liabilities=current_leases,
        operating_current_liabilities=op_cl,
        operating_nwc=operating_nwc,
        prev_operating_nwc=prev_nwc,
        change_in_operating_nwc=change_in_nwc,
    )

    # 5. Ratios
    # ROE
    roe = (net_income / total_equity) if total_equity > 0 else None

    # ROCE = EBIT / (Total Assets - Current Liabilities)
    capital_employed = total_assets - current_liabilities
    roce = (ebit / capital_employed) if capital_employed > 0 else None

    # Invested Capital = Total Equity + Debt + Leases - Cash
    invested_capital = total_equity + total_debt + total_leases - total_cash_liquid
    roic = (nopat / invested_capital) if invested_capital > 0 else None

    # Debt ratios
    debt_to_equity = (total_debt / total_equity) if total_equity > 0 else None
    total_debt_to_equity = ((total_debt + total_leases) / total_equity) if total_equity > 0 else None

    # Interest coverage
    interest_coverage = (ebit / interest_expense) if interest_expense > 0 else (999.0 if ebit > 0 else None)

    # FCF Yield
    fcf_yield = (fcf_reported / market_cap) if market_cap > 0 else None

    # EV / EBITDA
    ev_to_ebitda = (enterprise_value / ebitda) if (ebitda > 0 and enterprise_value > 0) else None

    # Margins
    operating_margin = (ebit / revenue) if revenue > 0 else None
    net_profit_margin = (net_income / revenue) if revenue > 0 else None
    asset_turnover = (revenue / total_assets) if total_assets > 0 else None

    return RatioMetrics(
        symbol=profile.symbol,
        company_name=profile.company_name,
        period_date=period_date,
        revenue=revenue,
        ebit=ebit,
        ebitda=ebitda,
        net_income=net_income,
        tax_expense=tax_expense,
        effective_tax_rate=effective_tax_rate,
        nopat=nopat,
        depreciation_and_amortization=dna,
        capex=capex,
        free_cash_flow_reported=fcf_reported,
        capital_structure=cap_struct,
        working_capital=working_cap,
        roe=roe,
        roce=roce,
        roic=roic,
        debt_to_equity=debt_to_equity,
        total_debt_to_equity=total_debt_to_equity,
        interest_coverage=interest_coverage,
        fcf_yield=fcf_yield,
        ev_to_ebitda=ev_to_ebitda,
        net_profit_margin=net_profit_margin,
        operating_margin=operating_margin,
        asset_turnover=asset_turnover,
        warnings=warnings,
    )
