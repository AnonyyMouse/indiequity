"""
Test fixtures and mock financial statements for indiequity unit tests.

Provides realistic, mathematically consistent Indian companies:
1. 'Bharat Precision Ltd' (Industrial Small-Cap)
2. 'Hindustan Commercial Bank' (Private Sector Bank)
"""

import pandas as pd
import pytest
from indiequity.data import FinancialProfile


@pytest.fixture
def mock_indian_smallcap() -> FinancialProfile:
    """
    Returns a FinancialProfile representing a high-quality Indian small-cap:
    'Bharat Precision Ltd' (Ticker: BHARATPRE.NS)
    """
    col_date = "2024-03-31"

    income_data = {
        "Total Revenue": [12_000_000_000.0],
        "Operating Income": [2_400_000_000.0],
        "EBIT": [2_400_000_000.0],
        "Interest Expense": [300_000_000.0],
        "Pretax Income": [2_100_000_000.0],
        "Tax Provision": [528_000_000.0],
        "Net Income Common Stockholders": [1_572_000_000.0],
    }
    income_stmt = pd.DataFrame(income_data, index=[col_date]).T

    balance_data = {
        "Total Assets": [10_000_000_000.0],
        "Current Assets": [4_000_000_000.0],
        "Current Liabilities": [2_000_000_000.0],
        "Stockholders Equity": [6_000_000_000.0],
        "Total Debt": [2_000_000_000.0],
        "Current Debt": [500_000_000.0],
        "Long Term Debt": [1_500_000_000.0],
        "Cash And Cash Equivalents": [500_000_000.0],
        "Other Short Term Investments": [0.0],
    }
    balance_sheet = pd.DataFrame(balance_data, index=[col_date]).T

    cashflow_data = {
        "Operating Cash Flow": [2_200_000_000.0],
        "Capital Expenditure": [-700_000_000.0],
        "Depreciation And Amortization": [400_000_000.0],
    }
    cash_flow = pd.DataFrame(cashflow_data, index=[col_date]).T

    info = {
        "currentPrice": 300.0,
        "marketCap": 15_000_000_000.0,
        "sharesOutstanding": 50_000_000.0,
        "enterpriseValue": 16_500_000_000.0,
        "beta": 1.15,
        "longName": "Bharat Precision Ltd",
        "sector": "Industrials",
        "industry": "Auto Components",
    }

    return FinancialProfile(
        symbol="BHARATPRE.NS",
        company_name="Bharat Precision Ltd",
        sector="Industrials",
        industry="Auto Components",
        current_price=300.0,
        market_cap=15_000_000_000.0,
        shares_outstanding=50_000_000.0,
        beta=1.15,
        income_stmt=income_stmt,
        balance_sheet=balance_sheet,
        cash_flow=cash_flow,
        latest_period_date="2024-03-31",
        available_periods=["2024-03-31"],
        raw_info=info,
    )


@pytest.fixture
def mock_indian_bank() -> FinancialProfile:
    """
    Returns a FinancialProfile representing an Indian private sector bank:
    'Hindustan Commercial Bank' (Ticker: HINDCOMM.NS)
    """
    col_date = "2024-03-31"

    income_data = {
        "Total Revenue": [25_000_000_000.0],
        "Operating Income": [8_000_000_000.0],
        "Pretax Income": [7_500_000_000.0],
        "Net Income Common Stockholders": [5_600_000_000.0],
    }
    income_stmt = pd.DataFrame(income_data, index=[col_date]).T

    balance_data = {
        "Total Assets": [200_000_000_000.0],
        "Stockholders Equity": [35_000_000_000.0],
        "Total Deposits": [160_000_000_000.0],
    }
    balance_sheet = pd.DataFrame(balance_data, index=[col_date]).T

    info = {
        "currentPrice": 750.0,
        "marketCap": 37_500_000_000.0,
        "sharesOutstanding": 50_000_000.0,
        "beta": 1.05,
        "longName": "Hindustan Commercial Bank Ltd",
        "sector": "Financial Services",
        "industry": "Banks - Regional",
    }

    return FinancialProfile(
        symbol="HINDCOMM.NS",
        company_name="Hindustan Commercial Bank Ltd",
        sector="Financial Services",
        industry="Banks - Regional",
        current_price=750.0,
        market_cap=37_500_000_000.0,
        shares_outstanding=50_000_000.0,
        beta=1.05,
        income_stmt=income_stmt,
        balance_sheet=balance_sheet,
        cash_flow=pd.DataFrame(),
        latest_period_date="2024-03-31",
        available_periods=["2024-03-31"],
        raw_info=info,
    )
