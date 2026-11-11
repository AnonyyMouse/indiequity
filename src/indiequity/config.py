"""
Configuration and constants for Indian financial markets.

Defines benchmark defaults, currency formatting in Crores (₹ Cr),
and ticker suffix rules for the National Stock Exchange (NSE) and
Bombay Stock Exchange (BSE).
"""

from typing import Union

# Indian Market Benchmark Defaults (Yields & Risk Premiums)
# Reference: Reserve Bank of India (RBI) 10-Year Government Securities (G-Sec) benchmark yield
DEFAULT_RISK_FREE_RATE: float = 0.070  # 7.0% (historical long-term Indian G-Sec average)

# Reference: Long-term historical Equity Risk Premium (ERP) for Indian equities (Nifty 50 over G-Sec)
DEFAULT_EQUITY_RISK_PREMIUM: float = 0.065  # 6.5%

# Long-term nominal terminal GDP growth estimate for India
DEFAULT_TERMINAL_GROWTH_RATE: float = 0.050  # 5.0%

# Effective corporate tax rate in India (under Section 115BAA of Income Tax Act)
DEFAULT_TAX_RATE: float = 0.2517  # 25.17% (22% base + 10% surcharge + 4% cess)

# Ticker suffix conventions for Indian exchanges
NSE_SUFFIX: str = ".NS"
BSE_SUFFIX: str = ".BO"

# 1 Crore = 10,000,000 (10 million)
CRORE_DIVISOR: float = 10_000_000.0


def format_crores(value: Union[int, float], decimals: int = 2) -> str:
    """
    Format a raw Indian Rupee (INR) amount into Crores (₹ Cr) with Indian comma grouping.

    Example:
        >>> format_crores(12505000000)
        '₹1,250.50 Cr'
        >>> format_crores(4500000)
        '₹0.45 Cr'

    Args:
        value: Amount in raw INR.
        decimals: Number of decimal places to retain.

    Returns:
        Formatted string in ₹ Crores.
    """
    if value is None:
        return "N/A"
    try:
        val_cr = float(value) / CRORE_DIVISOR
        sign = "-" if val_cr < 0 else ""
        abs_val = abs(val_cr)
        
        # Format integer part and decimal part
        formatted_base = f"{abs_val:.{decimals}f}"
        parts = formatted_base.split(".")
        int_part = parts[0]
        dec_part = f".{parts[1]}" if len(parts) > 1 else ""

        # Indian comma grouping logic (last 3 digits, then groups of 2)
        if len(int_part) <= 3:
            grouped = int_part
        else:
            last_three = int_part[-3:]
            remaining = int_part[:-3]
            # Group remaining into pairs of 2 digits from right to left
            pairs = []
            while remaining:
                pairs.append(remaining[-2:])
                remaining = remaining[:-2]
            grouped = ",".join(reversed(pairs)) + "," + last_three

        return f"{sign}₹{grouped}{dec_part} Cr"
    except (ValueError, TypeError):
        return "N/A"


def format_percentage(value: Union[int, float], decimals: int = 2) -> str:
    """
    Format a decimal as a percentage string.

    Example:
        >>> format_percentage(0.1852)
        '18.52%'
    """
    if value is None:
        return "N/A"
    try:
        val_pct = float(value) * 100.0
        return f"{val_pct:.{decimals}f}%"
    except (ValueError, TypeError):
        return "N/A"


def resolve_ticker(symbol: str, exchange: str = "NSE") -> str:
    """
    Ensure the ticker has the proper exchange suffix for Yahoo Finance.

    Args:
        symbol: Stock ticker (e.g., 'TATAMOTORS', 'INFY.NS', '500570')
        exchange: Target exchange ('NSE' or 'BSE'). Defaults to 'NSE'.

    Returns:
        Properly suffixed ticker (e.g., 'TATAMOTORS.NS' or '500570.BO').
    """
    clean = symbol.strip().upper()
    if clean.endswith(NSE_SUFFIX) or clean.endswith(BSE_SUFFIX):
        return clean
    
    if exchange.upper() == "BSE":
        return f"{clean}{BSE_SUFFIX}"
    return f"{clean}{NSE_SUFFIX}"
