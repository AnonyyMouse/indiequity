"""
Financial charting module for Indian equity research using Matplotlib.

Generates publication-grade charts:
1. DuPont 5-Stage decomposition bar chart
2. DCF Enterprise Value to Equity Value waterfall bridge
3. Valuation & profitability ratios summary
"""

from pathlib import Path
from typing import Optional
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for headless environments and servers
import matplotlib.pyplot as plt
import numpy as np

from indiequity.config import CRORE_DIVISOR
from indiequity.dcf import DCFResult
from indiequity.dupont import DuPont5Stage
from indiequity.ratios import RatioMetrics


def plot_dupont_5stage(d5: DuPont5Stage, save_path: Optional[str] = None) -> plt.Figure:
    """
    Generate a 5-Stage DuPont ROE Decomposition Bar Chart.

    Args:
        d5: DuPont5Stage object.
        save_path: Optional file path to save the chart as PNG.

    Returns:
        matplotlib Figure object.
    """
    labels = [
        "Tax Burden\n(NI / EBT)",
        "Interest Burden\n(EBT / EBIT)",
        "Operating Margin\n(EBIT / Rev)",
        "Asset Turnover\n(Rev / Assets)",
        "Equity Multiplier\n(Assets / Equity)",
    ]
    values = [
        d5.tax_burden,
        d5.interest_burden,
        d5.operating_margin,
        d5.asset_turnover,
        d5.equity_multiplier,
    ]

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    colors = ["#2b5c8f", "#3b75af", "#4f92d1", "#289672", "#e28743"]

    bars = ax.bar(labels, values, color=colors, width=0.55, edgecolor="#1c2d42", linewidth=1.2)

    # Add data labels
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}x" if height > 1 else f"{height * 100:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_title(
        f"DuPont 5-Stage ROE Breakdown: {d5.symbol}\nCalculated ROE = {d5.calculated_roe * 100:.2f}%",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    ax.set_ylabel("Factor Multiplier / Margin", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")

    return fig


def plot_dcf_waterfall(dcf: DCFResult, save_path: Optional[str] = None) -> plt.Figure:
    """
    Generate a DCF Enterprise Value to Equity Value Bridge Chart (in ₹ Crores).

    Args:
        dcf: DCFResult object.
        save_path: Optional file path to save the chart as PNG.

    Returns:
        matplotlib Figure object.
    """
    labels = ["PV of 5Y FCF", "PV of Terminal Val", "Enterprise Value", "+ Cash", "- Debt", "Implied Equity Val"]
    
    pv_5y_cr = dcf.sum_pv_fcff / CRORE_DIVISOR
    pv_tv_cr = dcf.pv_terminal_value / CRORE_DIVISOR
    ev_cr = dcf.enterprise_value / CRORE_DIVISOR
    cash_cr = dcf.cash_and_equivalents / CRORE_DIVISOR
    debt_cr = dcf.total_debt / CRORE_DIVISOR
    eq_val_cr = dcf.implied_equity_value / CRORE_DIVISOR

    values = [pv_5y_cr, pv_tv_cr, ev_cr, cash_cr, -debt_cr, eq_val_cr]
    colors = ["#2b5c8f", "#3b75af", "#1f4e78", "#289672", "#c0392b", "#16a085"]

    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=150)
    bars = ax.bar(labels, values, color=colors, width=0.55, edgecolor="#1c2d42", linewidth=1.2)

    for bar, val in zip(bars, values):
        height = bar.get_height()
        va = "bottom" if height >= 0 else "top"
        ax.annotate(
            f"₹{abs(val):,.1f} Cr",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4 if height >= 0 else -12),
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=9.5,
            fontweight="bold",
        )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(
        f"DCF Valuation Bridge: {dcf.symbol} ({dcf.company_name})\n"
        f"Intrinsic Value: ₹{dcf.intrinsic_value_per_share:,.2f} | Current Price: ₹{dcf.current_price:,.2f} "
        f"({dcf.upside_downside_pct * 100:+.1f}% Upside)",
        fontsize=12.5,
        fontweight="bold",
        pad=15,
    )
    ax.set_ylabel("Value in ₹ Crores", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")

    return fig
