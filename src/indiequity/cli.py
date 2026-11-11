"""
Command Line Interface (CLI) for indiequity using Typer and Rich.

Commands:
- indiequity quote <ticker>
- indiequity ratios <ticker>
- indiequity dupont <ticker>
- indiequity dcf <ticker>
- indiequity screen [--preset ...]
- indiequity export <ticker>
- indiequity chart <ticker>
"""

from typing import Optional
import typer
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from indiequity.charts import plot_dcf_waterfall, plot_dupont_5stage
from indiequity.config import (
    CRORE_DIVISOR,
    DEFAULT_TERMINAL_GROWTH_RATE,
    format_crores,
    format_percentage,
)
from indiequity.data import fetch_company_data
from indiequity.dcf import calculate_dcf
from indiequity.dupont import decompose_dupont_3stage, decompose_dupont_5stage
from indiequity.export import export_screener_to_excel, export_to_excel
from indiequity.ratios import calculate_ratios
from indiequity.screener import (
    NIFTY_50_TICKERS,
    NIFTY_SMALLCAP_PRESET,
    PRESET_DEEP_VALUE,
    PRESET_HIGH_COMPOUNDER,
    PRESET_QUALITY_SMALLCAP,
    StockScreener,
)

app = typer.Typer(
    name="indiequity",
    help="Indian Equity Research Toolkit for fundamental analysis, DuPont, screening, and DCF valuation.",
    add_completion=False,
)
console = Console()


@app.command()
def quote(symbol: str = typer.Argument(..., help="NSE or BSE stock ticker (e.g. 'TATAMOTORS' or 'ITC')")):
    """Display quick market profile and key figures for an Indian stock."""
    with console.status(f"[bold cyan]Fetching public financial data for {symbol}...[/bold cyan]"):
        profile = fetch_company_data(symbol)
        ratios = calculate_ratios(profile)

    content = f"""[bold white]Company:[/bold white] {profile.company_name} ({profile.symbol})
[bold white]Sector / Industry:[/bold white] {profile.sector} | {profile.industry}
[bold white]Current Share Price:[/bold white] ₹{profile.current_price:,.2f}
[bold white]Market Capitalization:[/bold white] {format_crores(ratios.market_cap)}
[bold white]Enterprise Value:[/bold white] {format_crores(ratios.enterprise_value)}
[bold white]Beta:[/bold white] {profile.beta if profile.beta is not None else 'N/A'}
[bold white]ROE / ROCE:[/bold white] {format_percentage(ratios.roe)} / {format_percentage(ratios.roce)}
[bold white]Debt to Equity:[/bold white] {f'{ratios.debt_to_equity:.2f}' if ratios.debt_to_equity is not None else 'N/A'}
[bold white]Free Cash Flow (TTM):[/bold white] {format_crores(ratios.free_cash_flow)}"""

    console.print(Panel(content, title=f"[bold green]🇮🇳 {profile.symbol} Profile[/bold green]", border_style="cyan"))


@app.command()
def ratios(symbol: str = typer.Argument(..., help="Stock ticker symbol")):
    """Compute and display comprehensive fundamental financial ratios."""
    with console.status(f"[bold cyan]Calculating ratios for {symbol}...[/bold cyan]"):
        profile = fetch_company_data(symbol)
        r = calculate_ratios(profile)

    table = Table(title=f"Fundamental Ratios: {profile.company_name} ({profile.symbol})", show_header=True, header_style="bold magenta")
    table.add_column("Financial Metric", style="cyan")
    table.add_column("Value (Indian Reporting)", style="bold white", justify="right")
    table.add_column("Benchmark / Standard", style="dim")

    table.add_row("Return on Capital Employed (ROCE)", format_percentage(r.roce), "> 18.0% (Quality Compounder)")
    table.add_row("Return on Equity (ROE)", format_percentage(r.roe), "> 15.0% (Good profitability)")
    table.add_row("Debt to Equity Ratio (D/E)", f"{r.debt_to_equity:.2f}" if r.debt_to_equity is not None else "N/A", "< 0.50 (Conservative leverage)")
    table.add_row("Interest Coverage Ratio", f"{r.interest_coverage:.1f}x" if r.interest_coverage is not None else "N/A", "> 3.0x (Solvency buffer)")
    table.add_row("Free Cash Flow Yield", format_percentage(r.fcf_yield), "> 3.0% (Strong cash generation)")
    table.add_row("EV / EBITDA", f"{r.ev_to_ebitda:.2f}x" if r.ev_to_ebitda is not None else "N/A", "< 15.0x (Reasonable multiple)")
    table.add_row("Operating Profit Margin", format_percentage(r.operating_margin), "Core business pricing power")
    table.add_row("Net Profit Margin", format_percentage(r.net_profit_margin), "Net margin after taxes & interest")
    table.add_row("Asset Turnover", f"{r.asset_turnover:.2f}x" if r.asset_turnover is not None else "N/A", "Revenue generated per ₹ of Assets")

    console.print(table)


@app.command()
def dupont(
    symbol: str = typer.Argument(..., help="Stock ticker symbol"),
    stages: int = typer.Option(5, "--stages", "-s", help="DuPont stages (3 or 5)"),
):
    """Decompose Return on Equity (ROE) into operational efficiency, turnover, and leverage."""
    with console.status(f"[bold cyan]Decomposing ROE for {symbol}...[/bold cyan]"):
        profile = fetch_company_data(symbol)

    if stages == 3:
        d3 = decompose_dupont_3stage(profile)
        table = Table(title=f"3-Stage DuPont ROE Decomposition: {profile.symbol}", header_style="bold blue")
        table.add_column("DuPont Stage", style="cyan")
        table.add_column("Formula", style="dim")
        table.add_column("Value", style="bold green", justify="right")

        table.add_row("1. Net Profit Margin", "Net Income / Revenue", format_percentage(d3.net_profit_margin))
        table.add_row("2. Asset Turnover", "Revenue / Total Assets", f"{d3.asset_turnover:.2f}x")
        table.add_row("3. Equity Multiplier", "Total Assets / Total Equity", f"{d3.equity_multiplier:.2f}x")
        table.add_row("Resulting Calculated ROE", "Product of (1 x 2 x 3)", format_percentage(d3.calculated_roe))
        console.print(table)
        console.print(f"[bold yellow]Diagnosis:[/bold yellow] {d3.interpretation}")
    else:
        d5 = decompose_dupont_5stage(profile)
        table = Table(title=f"5-Stage DuPont ROE Decomposition: {profile.symbol}", header_style="bold blue")
        table.add_column("DuPont Factor", style="cyan")
        table.add_column("Definition", style="dim")
        table.add_column("Multiplier / %", style="bold green", justify="right")

        table.add_row("1. Tax Burden", "Net Income / EBT (Tax efficiency)", f"{d5.tax_burden:.2f}x")
        table.add_row("2. Interest Burden", "EBT / EBIT (Interest drag)", f"{d5.interest_burden:.2f}x")
        table.add_row("3. Operating Margin", "EBIT / Revenue (Pricing power)", format_percentage(d5.operating_margin))
        table.add_row("4. Asset Turnover", "Revenue / Total Assets (Asset velocity)", f"{d5.asset_turnover:.2f}x")
        table.add_row("5. Financial Leverage", "Total Assets / Total Equity (Leverage)", f"{d5.equity_multiplier:.2f}x")
        table.add_row("Resulting Calculated ROE", "Product of 5 factors", format_percentage(d5.calculated_roe))
        console.print(table)
        console.print(f"[bold yellow]Diagnosis:[/bold yellow] {d5.interpretation}")


@app.command()
def dcf(
    symbol: str = typer.Argument(..., help="Stock ticker symbol"),
    growth: Optional[float] = typer.Option(None, "--growth", "-g", help="Annual FCF growth rate (e.g. 0.12 for 12%)"),
    wacc: Optional[float] = typer.Option(None, "--wacc", "-w", help="Discount rate (e.g. 0.12 for 12%)"),
    terminal_growth: float = typer.Option(DEFAULT_TERMINAL_GROWTH_RATE, "--terminal-growth", "-t", help="Perpetual growth rate"),
):
    """Run 5-Year FCFF DCF Valuation with a 2D sensitivity table (WACC vs. g)."""
    with console.status(f"[bold cyan]Running DCF valuation for {symbol}...[/bold cyan]"):
        profile = fetch_company_data(symbol)
        res = calculate_dcf(
            profile,
            forecast_years=5,
            fcf_growth_rate=growth,
            wacc=wacc,
            terminal_growth_rate=terminal_growth,
        )

    # Valuation summary table
    summary = Table(title=f"5-Year FCFF DCF Valuation: {profile.company_name} ({profile.symbol})", header_style="bold green")
    summary.add_column("Valuation Metric", style="cyan")
    summary.add_column("Value", style="bold white", justify="right")

    summary.add_row("Current Share Price", f"₹{res.current_price:,.2f}")
    summary.add_row("Intrinsic Value per Share", f"₹{res.intrinsic_value_per_share:,.2f}")
    upside_color = "green" if res.upside_downside_pct >= 0 else "red"
    summary.add_row("Implied Upside / Downside", f"[{upside_color}]{res.upside_downside_pct * 100:+.1f}%[/{upside_color}]")
    summary.add_row("Applied Discount Rate (WACC)", format_percentage(res.wacc))
    summary.add_row("Terminal Growth Rate (g)", format_percentage(res.terminal_growth_rate))
    summary.add_row("Present Value of 5Y Cash Flows", format_crores(res.sum_pv_fcff))
    summary.add_row("Present Value of Terminal Value", format_crores(res.pv_terminal_value))
    summary.add_row("Implied Enterprise Value", format_crores(res.enterprise_value))
    summary.add_row("Implied Equity Value", format_crores(res.implied_equity_value))
    console.print(summary)

    # Sensitivity grid table
    sens_table = Table(title="2D Sensitivity Matrix: Intrinsic Value per Share (₹)", header_style="bold magenta")
    sens_table.add_column("WACC \\ g", style="bold cyan")
    for col in res.sensitivity_table.columns:
        sens_table.add_column(str(col), justify="right")

    for idx, row in res.sensitivity_table.iterrows():
        row_vals = [f"₹{v:,.2f}" if v is not None and str(v) != "nan" else "N/A" for v in row]
        sens_table.add_row(str(idx), *row_vals)

    console.print(sens_table)


@app.command()
def screen(
    universe: str = typer.Option("smallcap", "--universe", "-u", help="'smallcap' or 'nifty50'"),
    preset: str = typer.Option("quality-smallcap", "--preset", "-p", help="'quality-smallcap', 'deep-value', or 'high-compounder'"),
    export: Optional[str] = typer.Option(None, "--export", "-e", help="Optional path to save Excel results (e.g. screener.xlsx)"),
):
    """Run fundamental stock screener across an Indian stock universe."""
    tickers = NIFTY_50_TICKERS if universe.lower() == "nifty50" else NIFTY_SMALLCAP_PRESET

    if preset.lower() == "deep-value":
        criteria = PRESET_DEEP_VALUE
    elif preset.lower() == "high-compounder":
        criteria = PRESET_HIGH_COMPOUNDER
    else:
        criteria = PRESET_QUALITY_SMALLCAP

    console.print(f"[bold cyan]🔍 Screening {len(tickers)} Indian stocks using '{preset}' criteria...[/bold cyan]")
    screener = StockScreener(criteria=criteria)

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task = progress.add_task("Screening stocks...", total=len(tickers))

        def on_progress(cur, total, sym):
            progress.update(task, description=f"Analyzing {sym} ({cur}/{total})...", advance=1)

        df = screener.screen(tickers, progress_callback=on_progress)

    # Display results table
    table = Table(title=f"Screener Results ({preset.upper()})", header_style="bold green")
    table.add_column("Symbol", style="cyan")
    table.add_column("Mcap (₹ Cr)", justify="right")
    table.add_column("ROCE (%)", justify="right")
    table.add_column("D/E", justify="right")
    table.add_column("Status", justify="center")
    table.add_column("Notes", style="dim")

    for _, row in df.iterrows():
        status = "[green]PASS[/green]" if row["Passed"] else "[red]FAIL[/red]"
        mcap = f"₹{row['Market Cap (₹ Cr)']:,.1f}" if row["Market Cap (₹ Cr)"] else "N/A"
        roce = f"{row['ROCE (%)']:.1f}%" if row["ROCE (%)"] is not None else "N/A"
        de = f"{row['D/E']:.2f}" if row["D/E"] is not None else "N/A"
        table.add_row(row["Symbol"], mcap, roce, de, status, row["Notes"])

    console.print(table)

    if export:
        export_screener_to_excel(df, export)
        console.print(f"[bold green]✅ Results saved to {export}[/bold green]")


@app.command()
def export(
    symbol: str = typer.Argument(..., help="Stock ticker symbol"),
    output: str = typer.Option("research_report.xlsx", "--output", "-o", help="Output .xlsx file path"),
):
    """Export complete equity research analysis into a formatted Excel workbook."""
    with console.status(f"[bold cyan]Generating Excel research report for {symbol}...[/bold cyan]"):
        profile = fetch_company_data(symbol)
        r = calculate_ratios(profile)
        d3 = decompose_dupont_3stage(profile)
        d5 = decompose_dupont_5stage(profile)
        dcf_res = calculate_dcf(profile)
        path = export_to_excel(profile, r, d3, d5, dcf_res, output)

    console.print(f"[bold green]✅ Research report successfully exported to:[/bold green] [bold cyan]{path}[/bold cyan]")


@app.command()
def chart(
    symbol: str = typer.Argument(..., help="Stock ticker symbol"),
    chart_type: str = typer.Option("dcf", "--type", "-t", help="'dcf' or 'dupont'"),
    output: str = typer.Option("chart.png", "--output", "-o", help="Output PNG path"),
):
    """Generate and save publication-grade financial charts."""
    with console.status(f"[bold cyan]Rendering {chart_type} chart for {symbol}...[/bold cyan]"):
        profile = fetch_company_data(symbol)
        if chart_type.lower() == "dupont":
            d5 = decompose_dupont_5stage(profile)
            plot_dupont_5stage(d5, save_path=output)
        else:
            dcf_res = calculate_dcf(profile)
            plot_dcf_waterfall(dcf_res, save_path=output)

    console.print(f"[bold green]✅ Chart successfully saved to:[/bold green] [bold cyan]{output}[/bold cyan]")


if __name__ == "__main__":
    app()
