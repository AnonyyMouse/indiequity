"""
Test fixtures and mock financial statements for indiequity unit tests.

Provides a realistic, mathematically consistent Indian small-cap company
("Bharat Precision Ltd") so tests run completely offline and deterministically.
"""

import pandas as pd
import pytest
from indiequity.data import FinancialProfile


@pytest.fixture
def mock_indian_smallcap() -> FinancialProfile:
    """
    Returns a FinancialProfile representing a high-quality Indian small-cap:
    'Bharat Precision Ltd' (Ticker: BHARATPRE.NS)

    Key Financials (in INR):
    - Market Cap: ₹1,500 Cr (15,000,000,000)
    - Revenue: ₹1,200 Cr (12,000,000,000)
    - EBIT: ₹240 Cr (2,400,000,000)
    - Interest Expense: ₹30 Cr (300,000,000)
    - EBT: ₹210 Cr (2,100,000,000)
    - Net Income: ₹157.2 Cr (1,572,000,000)
    - Total Assets: ₹1,000 Cr (10,000,000,000)
    - Current Liabilities: ₹200 Cr (2,000,000,000)
    - Stockholders' Equity: ₹600 Cr (6,000,000,000)
    - Total Debt: ₹200 Cr (2,000,000,000)
    - Cash & Cash Equivalents: ₹50 Cr (500,000,000)
    - Operating Cash Flow: ₹220 Cr (2,200,000,000)
    - CapEx: ₹70 Cr (700,000,000)
    - Depreciation & Amortization: ₹40 Cr (400,000,000)
    """
    col_date = "2024-03-31"

    income_data = {
        "Total Revenue": [12_000_000_000.0],
        "Operating Income": [2_400_000_000.0],
        "EBIT": [2_400_000_000.0],
        "Interest Expense": [300_000_000.0],
        "Pretax Income": [2_100_000_000.0],
        "Net Income Common Stockholders": [1_572_000_000.0],
    }
    income_stmt = pd.DataFrame(income_data, index=[col_date]).T

    balance_data = {
        "Total Assets": [10_000_000_000.0],
        "Current Liabilities": [2_000_000_000.0],
        "Stockholders Equity": [6_000_000_000.0],
        "Total Debt": [2_000_000_000.0],
        "Cash And Cash Equivalents": [500_000_000.0],
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
        raw_info=info,
    )
