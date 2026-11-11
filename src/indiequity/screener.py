"""
Stock Screener module for Indian equities (NSE and BSE).

Provides customizable fundamental screening filters and built-in
strategy presets (e.g., 'Quality Smallcap', 'Deep Value', 'High Compounder')
across predefined Indian stock universes.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional
import pandas as pd

from indiequity.config import CRORE_DIVISOR, format_crores, format_percentage
from indiequity.data import FinancialProfile, fetch_company_data
from indiequity.ratios import RatioMetrics, calculate_ratios

# --- Built-in Indian Stock Universes ---
NIFTY_50_TICKERS: List[str] = [
    "RELIANCE", "TCS", "HDFCBANK", "BHARTIARTL", "ICICIBANK",
    "INFY", "ITC", "LT", "HINDUNILVR", "SBIN",
    "TATAMOTORS", "SUNPHARMA", "BAJFINANCE", "MARUTI", "HCLTECH",
    "NTPC", "ONGC", "TITAN", "AXISBANK", "ADANIENT",
    "KOTAKBANK", "POWERGRID", "COALINDIA", "TATASTEEL", "ASIANPAINT",
    "BAJAJFINSV", "ULTRACEMCO", "WIPRO", "NESTLEIND", "JSWSTEEL",
    "GRASIM", "TECHM", "BAJAJ-AUTO", "HINDALCO", "CIPLA",
    "DRREDDY", "DIVISLAB", "APOLLOHOSP", "EICHERMOT", "BRITANNIA",
    "BPCL", "HEROMOTOCO", "INDUSINDBK", "SBILIFE", "HDFCLIFE",
    "TATACONSUM", "SHRIRAMFIN", "TRENT", "BEL", "ADANIPORTS"
]

NIFTY_SMALLCAP_PRESET: List[str] = [
    "CAMS", "CDSL", "RADICO", "ANGELONE", "SONACOMS",
    "KPITTECH", "DEEPAKNTR", "ROUTE", "KNRCON", "HAPPSTMNDS",
    "FINEORG", "CENTURYPLY", "METROPOLIS", "BSOFT", "KAJARIACER",
    "EQUITASBNK", "CREDITACC", "JBCHEPHARM", "BLUEDART", "PRAJIND"
]


@dataclass
class ScreenerCriteria:
    """Configurable quantitative filters for fundamental equity screening."""
    min_roce: Optional[float] = None              # e.g., 0.15 for 15%
    min_roe: Optional[float] = None               # e.g., 0.15 for 15%
    max_debt_to_equity: Optional[float] = None    # e.g., 0.5
    min_interest_coverage: Optional[float] = None # e.g., 3.0x
    min_fcf_yield: Optional[float] = None         # e.g., 0.03 for 3%
    max_ev_to_ebitda: Optional[float] = None      # e.g., 15.0x
    min_market_cap_cr: Optional[float] = None     # in ₹ Crores (e.g., 100.0)
    max_market_cap_cr: Optional[float] = None     # in ₹ Crores (e.g., 5000.0)
    min_operating_margin: Optional[float] = None  # e.g., 0.12 for 12%


# --- Industry Benchmark Presets ---
PRESET_QUALITY_SMALLCAP = ScreenerCriteria(
    min_roce=0.15,
    max_debt_to_equity=0.5,
    max_market_cap_cr=5000.0,
    min_interest_coverage=3.0,
)

PRESET_DEEP_VALUE = ScreenerCriteria(
    max_ev_to_ebitda=12.0,
    min_fcf_yield=0.04,
    max_debt_to_equity=0.8,
)

PRESET_HIGH_COMPOUNDER = ScreenerCriteria(
    min_roce=0.20,
    min_roe=0.18,
    max_debt_to_equity=0.4,
    min_operating_margin=0.15,
)


class StockScreener:
    """
    Executes multi-criteria fundamental screening across Indian stocks.
    """

    def __init__(self, criteria: Optional[ScreenerCriteria] = None):
        self.criteria = criteria or PRESET_QUALITY_SMALLCAP

    def evaluate(self, metrics: RatioMetrics) -> tuple[bool, List[str]]:
        """
        Evaluate a company's metrics against the active criteria.

        Returns:
            Tuple of (passed: bool, failure_reasons: list[str])
        """
        c = self.criteria
        failures = []

        mcap_cr = metrics.market_cap / CRORE_DIVISOR

        if c.min_roce is not None:
            if metrics.roce is None or metrics.roce < c.min_roce:
                val = f"{metrics.roce * 100:.1f}%" if metrics.roce is not None else "N/A"
                failures.append(f"ROCE {val} < {c.min_roce * 100:.1f}%")

        if c.min_roe is not None:
            if metrics.roe is None or metrics.roe < c.min_roe:
                val = f"{metrics.roe * 100:.1f}%" if metrics.roe is not None else "N/A"
                failures.append(f"ROE {val} < {c.min_roe * 100:.1f}%")

        if c.max_debt_to_equity is not None:
            if metrics.debt_to_equity is None or metrics.debt_to_equity > c.max_debt_to_equity:
                val = f"{metrics.debt_to_equity:.2f}" if metrics.debt_to_equity is not None else "N/A"
                failures.append(f"D/E {val} > {c.max_debt_to_equity:.2f}")

        if c.min_interest_coverage is not None:
            if metrics.interest_coverage is None or metrics.interest_coverage < c.min_interest_coverage:
                val = f"{metrics.interest_coverage:.1f}x" if metrics.interest_coverage is not None else "N/A"
                failures.append(f"Interest Cov {val} < {c.min_interest_coverage:.1f}x")

        if c.min_fcf_yield is not None:
            if metrics.fcf_yield is None or metrics.fcf_yield < c.min_fcf_yield:
                val = f"{metrics.fcf_yield * 100:.1f}%" if metrics.fcf_yield is not None else "N/A"
                failures.append(f"FCF Yield {val} < {c.min_fcf_yield * 100:.1f}%")

        if c.max_ev_to_ebitda is not None:
            if metrics.ev_to_ebitda is None or metrics.ev_to_ebitda > c.max_ev_to_ebitda:
                val = f"{metrics.ev_to_ebitda:.1f}x" if metrics.ev_to_ebitda is not None else "N/A"
                failures.append(f"EV/EBITDA {val} > {c.max_ev_to_ebitda:.1f}x")

        if c.min_market_cap_cr is not None:
            if mcap_cr < c.min_market_cap_cr:
                failures.append(f"Mcap ₹{mcap_cr:.1f} Cr < ₹{c.min_market_cap_cr:.1f} Cr")

        if c.max_market_cap_cr is not None:
            if mcap_cr > c.max_market_cap_cr:
                failures.append(f"Mcap ₹{mcap_cr:.1f} Cr > ₹{c.max_market_cap_cr:.1f} Cr")

        if c.min_operating_margin is not None:
            if metrics.operating_margin is None or metrics.operating_margin < c.min_operating_margin:
                val = f"{metrics.operating_margin * 100:.1f}%" if metrics.operating_margin is not None else "N/A"
                failures.append(f"Operating Margin {val} < {c.min_operating_margin * 100:.1f}%")

        passed = len(failures) == 0
        return passed, failures

    def screen(
        self,
        tickers: List[str],
        exchange: str = "NSE",
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> pd.DataFrame:
        """
        Execute screener across a list of tickers.

        Args:
            tickers: List of Indian ticker symbols.
            exchange: Default exchange ('NSE' or 'BSE').
            progress_callback: Optional callback func(current, total, ticker).

        Returns:
            pandas DataFrame containing all companies with their metrics and Passed flag.
        """
        records = []
        total = len(tickers)

        for i, sym in enumerate(tickers, start=1):
            if progress_callback:
                progress_callback(i, total, sym)

            try:
                profile = fetch_company_data(sym, exchange=exchange, use_cache=True)
                metrics = calculate_ratios(profile)
                passed, failures = self.evaluate(metrics)

                mcap_cr = metrics.market_cap / CRORE_DIVISOR
                records.append({
                    "Symbol": profile.symbol,
                    "Company": profile.company_name,
                    "Sector": profile.sector,
                    "Price (₹)": profile.current_price,
                    "Market Cap (₹ Cr)": round(mcap_cr, 2),
                    "ROCE (%)": round(metrics.roce * 100, 2) if metrics.roce is not None else None,
                    "ROE (%)": round(metrics.roe * 100, 2) if metrics.roe is not None else None,
                    "D/E": round(metrics.debt_to_equity, 2) if metrics.debt_to_equity is not None else None,
                    "Interest Cov (x)": round(metrics.interest_coverage, 2) if metrics.interest_coverage is not None else None,
                    "FCF Yield (%)": round(metrics.fcf_yield * 100, 2) if metrics.fcf_yield is not None else None,
                    "EV/EBITDA (x)": round(metrics.ev_to_ebitda, 2) if metrics.ev_to_ebitda is not None else None,
                    "Passed": passed,
                    "Notes": "; ".join(failures) if not passed else "Meets all criteria",
                })
            except Exception as e:
                records.append({
                    "Symbol": sym,
                    "Company": "Error fetching data",
                    "Sector": "N/A",
                    "Price (₹)": 0.0,
                    "Market Cap (₹ Cr)": 0.0,
                    "ROCE (%)": None,
                    "ROE (%)": None,
                    "D/E": None,
                    "Interest Cov (x)": None,
                    "FCF Yield (%)": None,
                    "EV/EBITDA (x)": None,
                    "Passed": False,
                    "Notes": str(e),
                })

        df = pd.DataFrame(records)
        return df
