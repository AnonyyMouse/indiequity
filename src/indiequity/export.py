"""
Data export module for Indian equity research (Excel and CSV).

Exports multi-tab, styled financial analysis workbooks (.xlsx) using openpyxl,
including summary ratios, DuPont breakdowns, DCF models, and sensitivity matrices.
"""

from pathlib import Path
from typing import Optional
import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
import pandas as pd

from indiequity.config import CRORE_DIVISOR, format_crores, format_percentage
from indiequity.data import FinancialProfile
from indiequity.dcf import DCFResult
from indiequity.dupont import DuPont3Stage, DuPont5Stage
from indiequity.ratios import RatioMetrics


def export_to_excel(
    profile: FinancialProfile,
    ratios: RatioMetrics,
    dupont3: DuPont3Stage,
    dupont5: DuPont5Stage,
    dcf: Optional[DCFResult],
    output_path: str,
) -> str:
    """
    Export comprehensive equity research report into a multi-tab Excel workbook.

    Worksheet tabs:
    1. Overview & Ratios
    2. DuPont Analysis
    3. DCF Valuation & Sensitivity
    4. Financial Statements

    Args:
        profile: Company FinancialProfile.
        ratios: RatioMetrics container.
        dupont3: DuPont3Stage container.
        dupont5: DuPont5Stage container.
        dcf: Optional DCFResult container.
        output_path: Destination path (.xlsx).

    Returns:
        Absolute path to the created Excel file.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    wb = openpyxl.Workbook()

    # Styling constants
    header_fill = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
    header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    title_font = Font(name="Calibri", size=14, bold=True, color="1F4E78")
    bold_font = Font(name="Calibri", size=11, bold=True)
    regular_font = Font(name="Calibri", size=11)
    border = Border(
        left=Side(style="thin", color="D9D9D9"),
        right=Side(style="thin", color="D9D9D9"),
        top=Side(style="thin", color="D9D9D9"),
        bottom=Side(style="thin", color="D9D9D9"),
    )

    # --- TAB 1: Overview & Ratios ---
    ws_ratios = wb.active
    ws_ratios.title = "Overview & Ratios"

    ws_ratios["A1"] = f"{profile.company_name} ({profile.symbol})"
    ws_ratios["A1"].font = title_font
    ws_ratios["A2"] = f"Sector: {profile.sector} | Industry: {profile.industry}"
    ws_ratios["A2"].font = regular_font

    ratios_data = [
        ("Market Capitalization", format_crores(ratios.market_cap)),
        ("Current Share Price", f"₹{profile.current_price:,.2f}"),
        ("Enterprise Value", format_crores(ratios.enterprise_value)),
        ("Total Revenue", format_crores(ratios.revenue)),
        ("Operating Profit (EBIT)", format_crores(ratios.ebit)),
        ("EBITDA", format_crores(ratios.ebitda)),
        ("Net Income", format_crores(ratios.net_income)),
        ("Free Cash Flow (FCF)", format_crores(ratios.free_cash_flow)),
        ("---", "---"),
        ("Return on Capital Employed (ROCE)", format_percentage(ratios.roce)),
        ("Return on Equity (ROE)", format_percentage(ratios.roe)),
        ("Debt to Equity Ratio (D/E)", f"{ratios.debt_to_equity:.2f}" if ratios.debt_to_equity is not None else "N/A"),
        ("Interest Coverage Ratio", f"{ratios.interest_coverage:.1f}x" if ratios.interest_coverage is not None else "N/A"),
        ("FCF Yield", format_percentage(ratios.fcf_yield)),
        ("EV / EBITDA", f"{ratios.ev_to_ebitda:.2f}x" if ratios.ev_to_ebitda is not None else "N/A"),
        ("Operating Margin", format_percentage(ratios.operating_margin)),
        ("Net Profit Margin", format_percentage(ratios.net_profit_margin)),
        ("Asset Turnover", f"{ratios.asset_turnover:.2f}x" if ratios.asset_turnover is not None else "N/A"),
    ]

    ws_ratios.append([])
    ws_ratios.append(["Metric / Financial Item", "Value (Indian Reporting Standard)"])
    header_row = ws_ratios.max_row
    for col in range(1, 3):
        cell = ws_ratios.cell(row=header_row, column=col)
        cell.fill = header_fill
        cell.font = header_font

    for item, val in ratios_data:
        ws_ratios.append([item, val])
        r = ws_ratios.max_row
        ws_ratios.cell(row=r, column=1).font = regular_font
        ws_ratios.cell(row=r, column=2).font = bold_font

    # --- TAB 2: DuPont Analysis ---
    ws_dupont = wb.create_sheet(title="DuPont Analysis")
    ws_dupont["A1"] = f"DuPont ROE Decomposition: {profile.symbol}"
    ws_dupont["A1"].font = title_font

    ws_dupont.append([])
    ws_dupont.append(["3-Stage DuPont Component", "Value", "Meaning"])
    h_row = ws_dupont.max_row
    for col in range(1, 4):
        ws_dupont.cell(row=h_row, column=col).fill = header_fill
        ws_dupont.cell(row=h_row, column=col).font = header_font

    d3_rows = [
        ("Net Profit Margin", f"{dupont3.net_profit_margin * 100:.2f}%", "Operating & Pricing Profitability"),
        ("Asset Turnover", f"{dupont3.asset_turnover:.2f}x", "Asset Efficiency in generating Revenue"),
        ("Equity Multiplier", f"{dupont3.equity_multiplier:.2f}x", "Financial Leverage (Assets / Equity)"),
        ("Resulting ROE (Product)", f"{dupont3.calculated_roe * 100:.2f}%", dupont3.interpretation),
    ]
    for r in d3_rows:
        ws_dupont.append(list(r))

    ws_dupont.append([])
    ws_dupont.append(["5-Stage DuPont Component", "Value", "Meaning"])
    h_row2 = ws_dupont.max_row
    for col in range(1, 4):
        ws_dupont.cell(row=h_row2, column=col).fill = header_fill
        ws_dupont.cell(row=h_row2, column=col).font = header_font

    d5_rows = [
        ("Tax Burden (NI / EBT)", f"{dupont5.tax_burden:.2f}x", "Percentage of pre-tax profit kept after taxes"),
        ("Interest Burden (EBT / EBIT)", f"{dupont5.interest_burden:.2f}x", "Operating profit retained after debt costs"),
        ("Operating Margin (EBIT / Rev)", f"{dupont5.operating_margin * 100:.2f}%", "Core business operating efficiency"),
        ("Asset Turnover (Rev / Assets)", f"{dupont5.asset_turnover:.2f}x", "Asset utilization rate"),
        ("Financial Leverage (Assets / Equity)", f"{dupont5.equity_multiplier:.2f}x", "Capital structure leverage"),
        ("Resulting ROE (Product)", f"{dupont5.calculated_roe * 100:.2f}%", dupont5.interpretation),
    ]
    for r in d5_rows:
        ws_dupont.append(list(r))

    # --- TAB 3: DCF Valuation ---
    if dcf is not None:
        ws_dcf = wb.create_sheet(title="DCF Valuation")
        ws_dcf["A1"] = f"5-Year FCFF DCF Valuation: {profile.symbol}"
        ws_dcf["A1"].font = title_font

        ws_dcf.append([])
        ws_dcf.append(["Valuation Metric", "Value"])
        h = ws_dcf.max_row
        for col in range(1, 3):
            ws_dcf.cell(row=h, column=col).fill = header_fill
            ws_dcf.cell(row=h, column=col).font = header_font

        dcf_summary = [
            ("Current Market Price", f"₹{dcf.current_price:,.2f}"),
            ("Intrinsic Value per Share", f"₹{dcf.intrinsic_value_per_share:,.2f}"),
            ("Upside / Downside (%)", f"{dcf.upside_downside_pct * 100:+.2f}%"),
            ("Discount Rate (WACC)", format_percentage(dcf.wacc)),
            ("Terminal Perpetual Growth (g)", format_percentage(dcf.terminal_growth_rate)),
            ("Present Value of 5Y Cash Flows", format_crores(dcf.sum_pv_fcff)),
            ("Present Value of Terminal Value", format_crores(dcf.pv_terminal_value)),
            ("Implied Enterprise Value", format_crores(dcf.enterprise_value)),
            ("Cash & Cash Equivalents", format_crores(dcf.cash_and_equivalents)),
            ("Total Debt", format_crores(dcf.total_debt)),
            ("Implied Equity Value", format_crores(dcf.implied_equity_value)),
        ]
        for item, val in dcf_summary:
            ws_dcf.append([item, val])

        # Add Sensitivity Grid
        ws_dcf.append([])
        ws_dcf.append(["2D Sensitivity Matrix: Intrinsic Value per Share (₹)"])
        ws_dcf.cell(row=ws_dcf.max_row, column=1).font = bold_font

        # Write sensitivity table DataFrame
        sens = dcf.sensitivity_table
        ws_dcf.append(["WACC \\ g"] + list(sens.columns))
        grid_head = ws_dcf.max_row
        for c in range(1, len(sens.columns) + 2):
            ws_dcf.cell(row=grid_head, column=c).fill = header_fill
            ws_dcf.cell(row=grid_head, column=c).font = header_font

        for idx, row in sens.iterrows():
            ws_dcf.append([str(idx)] + [f"₹{val:,.2f}" if pd.notna(val) else "N/A" for val in row])

    # Auto-adjust column widths for all sheets
    for ws in wb.worksheets:
        for col in ws.columns:
            max_len = max(len(str(cell.value or "")) for cell in col)
            col_letter = openpyxl.utils.get_column_letter(col[0].column)
            ws.column_dimensions[col_letter].width = max(max_len + 3, 14)

    wb.save(output_path)
    return str(Path(output_path).resolve())


def export_screener_to_excel(screener_df: pd.DataFrame, output_path: str) -> str:
    """Export screener results table into a formatted Excel file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        screener_df.to_excel(writer, sheet_name="Screener Results", index=False)
    return str(Path(output_path).resolve())


def export_to_csv(data: pd.DataFrame, output_path: str) -> str:
    """Export any tabular DataFrame to CSV."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)
    return str(Path(output_path).resolve())
