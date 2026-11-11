"""
Data export module for Indian equity research (Excel and CSV).

Exports multi-tab, styled financial analysis workbooks (.xlsx) using openpyxl,
including actual sorted financial statements, capital structure reconciliation,
detailed forecast schedules, DuPont breakdowns, DCF models, and sensitivity matrices.
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
    3. DCF Valuation & Bridge
    4. Explicit Forecast Schedule
    5. Balance Sheet & Statements
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    wb = openpyxl.Workbook()

    # Styling constants
    header_fill = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
    subhead_fill = PatternFill(start_color="2B5C8F", end_color="2B5C8F", fill_type="solid")
    header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    title_font = Font(name="Calibri", size=14, bold=True, color="1F4E78")
    bold_font = Font(name="Calibri", size=11, bold=True)
    regular_font = Font(name="Calibri", size=11)

    # --- TAB 1: Overview & Ratios ---
    ws_ratios = wb.active
    ws_ratios.title = "Overview & Ratios"

    ws_ratios["A1"] = f"{profile.company_name} ({profile.symbol})"
    ws_ratios["A1"].font = title_font
    ws_ratios["A2"] = f"Sector: {profile.sector} | Industry: {profile.industry} | Latest Period: {profile.latest_period_date}"
    ws_ratios["A2"].font = regular_font

    cs = ratios.capital_structure
    wc = ratios.working_capital

    ratios_data = [
        ("Market Capitalization", format_crores(cs.market_cap)),
        ("Current Share Price", f"₹{profile.current_price:,.2f}"),
        ("Enterprise Value", format_crores(cs.enterprise_value)),
        ("Total Revenue", format_crores(ratios.revenue)),
        ("Operating Profit (EBIT)", format_crores(ratios.ebit)),
        ("NOPAT (EBIT x (1-t))", format_crores(ratios.nopat)),
        ("Net Income", format_crores(ratios.net_income)),
        ("---", "---"),
        ("Return on Invested Capital (ROIC)", format_percentage(ratios.roic)),
        ("Return on Capital Employed (ROCE)", format_percentage(ratios.roce)),
        ("Return on Equity (ROE)", format_percentage(ratios.roe)),
        ("Interest-Bearing Debt to Equity", f"{ratios.debt_to_equity:.2f}" if ratios.debt_to_equity is not None else "N/A"),
        ("Total Debt + Leases to Equity", f"{ratios.total_debt_to_equity:.2f}" if ratios.total_debt_to_equity is not None else "N/A"),
        ("Interest Coverage Ratio", f"{ratios.interest_coverage:.1f}x" if ratios.interest_coverage is not None else "N/A"),
        ("Operating Working Capital (NWC)", format_crores(wc.operating_nwc)),
        ("Change in Operating NWC (ΔNWC)", format_crores(wc.change_in_operating_nwc)),
        ("FCF Yield (Reported)", format_percentage(ratios.fcf_yield)),
        ("EV / EBITDA", f"{ratios.ev_to_ebitda:.2f}x" if ratios.ev_to_ebitda is not None else "N/A"),
        ("Operating Margin", format_percentage(ratios.operating_margin)),
        ("Net Profit Margin", format_percentage(ratios.net_profit_margin)),
    ]

    ws_ratios.append([])
    ws_ratios.append(["Metric / Financial Item", "Value (Indian Reporting Standard)"])
    h_row = ws_ratios.max_row
    for col in range(1, 3):
        ws_ratios.cell(row=h_row, column=col).fill = header_fill
        ws_ratios.cell(row=h_row, column=col).font = header_font

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
    h = ws_dupont.max_row
    for col in range(1, 4):
        ws_dupont.cell(row=h, column=col).fill = header_fill
        ws_dupont.cell(row=h, column=col).font = header_font

    for r in [
        ("Net Profit Margin", f"{dupont3.net_profit_margin * 100:.2f}%", "Operating & Pricing Profitability"),
        ("Asset Turnover", f"{dupont3.asset_turnover:.2f}x", "Asset Efficiency in generating Revenue"),
        ("Equity Multiplier", f"{dupont3.equity_multiplier:.2f}x", "Financial Leverage (Assets / Equity)"),
        ("Resulting ROE (Product)", f"{dupont3.calculated_roe * 100:.2f}%", dupont3.interpretation),
    ]:
        ws_dupont.append(list(r))

    ws_dupont.append([])
    ws_dupont.append(["5-Stage DuPont Component", "Value", "Meaning"])
    h2 = ws_dupont.max_row
    for col in range(1, 4):
        ws_dupont.cell(row=h2, column=col).fill = header_fill
        ws_dupont.cell(row=h2, column=col).font = header_font

    for r in [
        ("Tax Burden (NI / EBT)", f"{dupont5.tax_burden:.2f}x", "Portion of pre-tax profit retained after tax"),
        ("Interest Burden (EBT / EBIT)", f"{dupont5.interest_burden:.2f}x", "Portion of operating profit retained after debt cost"),
        ("Operating Margin (EBIT / Rev)", f"{dupont5.operating_margin * 100:.2f}%", "Pricing power and operational efficiency"),
        ("Asset Turnover (Rev / Assets)", f"{dupont5.asset_turnover:.2f}x", "Asset utilization velocity"),
        ("Financial Leverage (Assets / Equity)", f"{dupont5.equity_multiplier:.2f}x", "Balance sheet leverage multiple"),
        ("Resulting ROE (Product)", f"{dupont5.calculated_roe * 100:.2f}%", dupont5.interpretation),
    ]:
        ws_dupont.append(list(r))

    # --- TAB 3: DCF Valuation & Bridge ---
    if dcf is not None:
        ws_dcf = wb.create_sheet(title="DCF Valuation")
        ws_dcf["A1"] = f"{dcf.model_type}: {profile.symbol}"
        ws_dcf["A1"].font = title_font

        b = dcf.bridge
        ws_dcf.append([])
        ws_dcf.append(["Enterprise-to-Equity Bridge Component", "Amount"])
        h = ws_dcf.max_row
        for col in range(1, 3):
            ws_dcf.cell(row=h, column=col).fill = header_fill
            ws_dcf.cell(row=h, column=col).font = header_font

        bridge_rows = [
            ("Present Value of Explicit Cash Flows", format_crores(b.sum_pv_fcff)),
            (f"Present Value of Terminal Value ({b.terminal_value_pct_of_ev * 100:.1f}% of EV)", format_crores(b.pv_terminal_value)),
            ("Implied Enterprise Value", format_crores(b.enterprise_value)),
            ("(+) Total Liquid Cash & Short-Term Investments", format_crores(b.total_liquid_assets)),
            ("(-) Total Interest-Bearing Debt", format_crores(-b.total_interest_bearing_debt)),
            ("(-) Capital Lease Obligations", format_crores(-b.lease_liabilities)),
            ("(-) Non-Controlling (Minority) Interests", format_crores(-b.minority_interest)),
            ("Implied Equity Value", format_crores(b.implied_equity_value)),
            ("Diluted Shares Outstanding", f"{b.diluted_shares / 10_000_000:,.2f} Cr shares"),
            ("Intrinsic Value per Share", f"₹{b.intrinsic_value_per_share:,.2f}"),
            ("Current Market Price", f"₹{b.current_share_price:,.2f}"),
            ("Margin of Safety (Implied Upside/Downside)", f"{b.margin_of_safety_pct * 100:+.2f}%"),
            ("Applied WACC", format_percentage(dcf.wacc)),
            ("Cost of Equity (Ke)", format_percentage(dcf.cost_of_equity)),
            ("Terminal Growth Rate (g)", format_percentage(dcf.terminal_growth_rate)),
            ("Terminal ROIC", format_percentage(dcf.terminal_roic)),
            ("Terminal Reinvestment Rate", format_percentage(dcf.terminal_reinvestment_rate)),
        ]
        for item, val in bridge_rows:
            ws_dcf.append([item, val])

        # Sensitivity Table
        ws_dcf.append([])
        ws_dcf.append(["2D Sensitivity Matrix: Intrinsic Value per Share (₹)"])
        ws_dcf.cell(row=ws_dcf.max_row, column=1).font = bold_font

        sens = dcf.sensitivity_table
        ws_dcf.append(["WACC \\ g"] + list(sens.columns))
        grid_head = ws_dcf.max_row
        for c in range(1, len(sens.columns) + 2):
            ws_dcf.cell(row=grid_head, column=c).fill = header_fill
            ws_dcf.cell(row=grid_head, column=c).font = header_font

        for idx, row in sens.iterrows():
            ws_dcf.append([str(idx)] + [f"₹{val:,.2f}" if pd.notna(val) else "N/A" for val in row])

        # --- TAB 4: Explicit Forecast Schedule ---
        ws_sched = wb.create_sheet(title="Forecast Schedule")
        ws_sched["A1"] = f"Explicit Forecast Schedule: {profile.symbol}"
        ws_sched["A1"].font = title_font

        ws_sched.append([])
        headers = [
            "Year", "Revenue (₹ Cr)", "Growth", "Operating Margin",
            "EBIT (₹ Cr)", "Taxes (₹ Cr)", "NOPAT (₹ Cr)", "D&A (₹ Cr)",
            "CapEx (₹ Cr)", "ΔNWC (₹ Cr)", "FCFF (₹ Cr)", "Reinvestment",
            "Discount Factor", "PV FCFF (₹ Cr)"
        ]
        ws_sched.append(headers)
        h = ws_sched.max_row
        for c in range(1, len(headers) + 1):
            ws_sched.cell(row=h, column=c).fill = header_fill
            ws_sched.cell(row=h, column=c).font = header_font

        for f in dcf.annual_forecasts:
            ws_sched.append([
                f"Year {f.year}",
                round(f.revenue / CRORE_DIVISOR, 2),
                f"{f.revenue_growth * 100:.1f}%",
                f"{f.operating_margin * 100:.1f}%",
                round(f.ebit / CRORE_DIVISOR, 2),
                round(f.operating_taxes / CRORE_DIVISOR, 2),
                round(f.nopat / CRORE_DIVISOR, 2),
                round(f.depreciation_and_amortization / CRORE_DIVISOR, 2),
                round(f.capex / CRORE_DIVISOR, 2),
                round(f.change_in_nwc / CRORE_DIVISOR, 2),
                round(f.fcff / CRORE_DIVISOR, 2),
                round(f.reinvestment / CRORE_DIVISOR, 2),
                round(f.discount_factor, 4),
                round(f.pv_fcff / CRORE_DIVISOR, 2),
            ])

    # Auto-adjust column widths
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
