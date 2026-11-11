"""
Command Line Interface (CLI) for indiequity using Typer and Rich.

Commands:
- indiequity quote <ticker>
- indiequity ratios <ticker>
- indiequity dupont <ticker>
- indiequity dcf <ticker> [--growth ...] [--wacc ...] [--scenarios] [--reverse]
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
from indiequity.dcf import (
    ForecastAssumptions,
    InvalidValuationInputError,
    UnsupportedValuationModelError,
    calculate_dcf,
    calculate_residual_income_valuation,
    reverse_dcf,
    run_scenario_analysis,
)
from indiequity.dupont import decompose_dupont_3stage, decompose_dupont_5stage
from indiequity.piotroski import calculate_piotroski_score
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


def version_callback(value: bool):
    """Callback to print version and exit."""
    if value:
        import indiequity
        rprint(f"[bold cyan]indiequity[/bold cyan] version [bold green]{indiequity.__version__}[/bold green]")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-v",
        help="Show indiequity version and exit.",
        callback=version_callback,
        is_eager=True,
    ),
):
    """Indian Equity Research Toolkit for fundamental analysis, DuPont, screening, and DCF valuation."""
    pass


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
[bold white]ROE / ROCE / ROIC:[/bold white] {format_percentage(ratios.roe)} / {format_percentage(ratios.roce)} / {format_percentage(ratios.roic)}
[bold white]Debt to Equity:[/bold white] {f'{ratios.debt_to_equity:.2f}' if ratios.debt_to_equity is not None else 'N/A'} (Total with Leases: {f'{ratios.total_debt_to_equity:.2f}' if ratios.total_debt_to_equity is not None else 'N/A'})
[bold white]Free Cash Flow (Reported):[/bold white] {format_crores(ratios.free_cash_flow_reported)}"""

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

    table.add_row("Return on Invested Capital (ROIC)", format_percentage(r.roic), "> 15.0% (Economic Value Added)")
    table.add_row("Return on Capital Employed (ROCE)", format_percentage(r.roce), "> 18.0% (Quality Compounder)")
    table.add_row("Return on Equity (ROE)", format_percentage(r.roe), "> 15.0% (Shareholder profitability)")
    table.add_row("Interest-Bearing Debt to Equity", f"{r.debt_to_equity:.2f}" if r.debt_to_equity is not None else "N/A", "< 0.50 (Conservative leverage)")
    table.add_row("Total Debt + Leases to Equity", f"{r.total_debt_to_equity:.2f}" if r.total_debt_to_equity is not None else "N/A", "Includes lease obligations")
    table.add_row("Interest Coverage Ratio", f"{r.interest_coverage:.1f}x" if r.interest_coverage is not None else "N/A", "> 3.0x (Solvency buffer)")
    table.add_row("Operating Working Capital (NWC)", format_crores(r.working_capital.operating_nwc), "Excludes cash and borrowings")
    table.add_row("Change in Operating NWC (ΔNWC)", format_crores(r.working_capital.change_in_operating_nwc), "Annual cash outflow into working capital")
    table.add_row("Free Cash Flow Yield", format_percentage(r.fcf_yield), "> 3.0% (Strong cash generation)")
    table.add_row("EV / EBITDA", f"{r.ev_to_ebitda:.2f}x" if r.ev_to_ebitda is not None else "N/A", "< 15.0x (Reasonable multiple)")
    table.add_row("Operating Profit Margin", format_percentage(r.operating_margin), "Core business pricing power")
    table.add_row("Net Profit Margin", format_percentage(r.net_profit_margin), "Net margin after taxes & interest")

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
def piotroski(symbol: str = typer.Argument(..., help="Stock ticker symbol")):
    """Compute the 9-point Piotroski F-Score (financial health) for a stock."""
    with console.status(f"[bold cyan]Calculating Piotroski F-Score for {symbol}...[/bold cyan]"):
        profile = fetch_company_data(symbol)
        p = calculate_piotroski_score(profile)

    def mark(passed: bool) -> str:
        return "[bold green]PASS (1)[/bold green]" if passed else "[bold red]FAIL (0)[/bold red]"

    table = Table(title=f"Piotroski F-Score: {profile.company_name} ({profile.symbol})", header_style="bold blue")
    table.add_column("Category", style="dim")
    table.add_column("Signal", style="cyan")
    table.add_column("Test", style="dim")
    table.add_column("Result", justify="right")

    table.add_row("Profitability", "1. ROA Positive", f"ROA = {format_percentage(p.roa)}", mark(p.roa_positive))
    table.add_row("Profitability", "2. Operating Cash Flow Positive", "CFO > 0", mark(p.cfo_positive))
    table.add_row("Profitability", "3. ROA Improving", f"{format_percentage(p.prev_roa)} -> {format_percentage(p.roa)}", mark(p.roa_improved))
    table.add_row("Profitability", "4. Accrual Quality", "CFO > Net Income", mark(p.accrual_quality))
    table.add_row("Leverage", "5. Leverage Decreasing", f"LT Debt/Assets {p.prev_leverage_ratio:.2f} -> {p.leverage_ratio:.2f}", mark(p.leverage_decreased))
    table.add_row("Leverage", "6. Liquidity Improving", f"Current Ratio {p.prev_current_ratio:.2f} -> {p.current_ratio:.2f}", mark(p.liquidity_improved))
    table.add_row("Leverage", "7. No Dilution", "Shares outstanding not increased", mark(p.no_dilution))
    table.add_row("Efficiency", "8. Gross Margin Improving", f"{format_percentage(p.prev_gross_margin)} -> {format_percentage(p.gross_margin)}", mark(p.gross_margin_improved))
    table.add_row("Efficiency", "9. Asset Turnover Improving", f"{p.prev_asset_turnover:.2f}x -> {p.asset_turnover:.2f}x", mark(p.asset_turnover_improved))
    table.add_row("", "[bold]Piotroski F-Score[/bold]", "Out of 9", f"[bold]{p.score} / 9[/bold]")

    console.print(table)
    console.print(f"[bold yellow]Diagnosis:[/bold yellow] {p.interpretation}")


@app.command()
def dcf(
    symbol: str = typer.Argument(..., help="Stock ticker symbol"),
    growth: Optional[float] = typer.Option(None, "--growth", "-g", help="Annual growth rate (e.g. 0.12 for 12%)"),
    wacc: Optional[float] = typer.Option(None, "--wacc", "-w", help="Discount rate override (e.g. 0.12 for 12%)"),
    terminal_growth: float = typer.Option(DEFAULT_TERMINAL_GROWTH_RATE, "--terminal-growth", "-t", help="Perpetual growth rate"),
    years: int = typer.Option(5, "--years", "-y", help="Forecast years (5 or 10)"),
    scenarios: bool = typer.Option(False, "--scenarios", help="Run Bear, Base, and Bull valuation scenarios"),
    reverse: bool = typer.Option(False, "--reverse", help="Run Reverse DCF solving for market-implied growth"),
):
    """Run FCFF DCF Valuation with explicit forecasts, WACC, and 2D sensitivity table."""
    with console.status(f"[bold cyan]Analyzing {symbol}...[/bold cyan]"):
        profile = fetch_company_data(symbol)

    if profile.is_financial_institution():
        console.print(f"[bold yellow]⚠️ {profile.company_name} is a Financial Institution ({profile.sector} / {profile.industry}).[/bold yellow]")
        console.print("[dim]Industrial FCFF/WACC DCF cannot be applied to banks/NBFCs because debt is operating funding. Running Residual Income Model (RIM)...[/dim]")
        rim = calculate_residual_income_valuation(profile)

        table = Table(title=f"Residual Income Valuation: {profile.company_name} ({profile.symbol})", header_style="bold blue")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="bold white", justify="right")
        table.add_row("Current Share Price", f"₹{rim.current_share_price:,.2f}")
        table.add_row("Book Value per Share", f"₹{rim.book_value_per_share:,.2f}")
        table.add_row("Normalized ROE", format_percentage(rim.normalized_roe))
        table.add_row("Cost of Equity (Ke)", format_percentage(rim.cost_of_equity))
        table.add_row("Sustainable Growth (g)", format_percentage(rim.sustainable_growth_rate))
        table.add_row("Justified P/B Multiple", f"{rim.justified_pb_ratio:.2f}x")
        table.add_row("Justified Share Price", f"₹{rim.justified_price_per_share:,.2f}")
        table.add_row("Intrinsic Value (RIM)", f"₹{rim.intrinsic_value_per_share:,.2f}")
        table.add_row("Margin of Safety", f"{rim.margin_of_safety_pct * 100:+.1f}%")
        console.print(table)
        return

    if reverse:
        res = reverse_dcf(profile, wacc=wacc, terminal_growth_rate=terminal_growth)
        rev_table = Table(title=f"Reverse DCF: Market-Implied Expectations for {profile.symbol}", header_style="bold magenta")
        rev_table.add_column("Metric", style="cyan")
        rev_table.add_column("Value", style="bold white", justify="right")
        rev_table.add_row("Current Market Price", f"₹{res['market_price']:,.2f}")
        rev_table.add_row("Implied 5-Year Revenue CAGR", str(res["implied_5y_cagr_pct"]))
        rev_table.add_row("Market Expectation Assessment", str(res["market_expectation_assessment"]))
        console.print(rev_table)
        return

    if scenarios:
        scen_results = run_scenario_analysis(profile, base_growth=growth)
        s_table = Table(title=f"Scenario Valuation Analysis: {profile.symbol}", header_style="bold green")
        s_table.add_column("Scenario", style="bold cyan")
        s_table.add_column("WACC", justify="right")
        s_table.add_column("Term Growth (g)", justify="right")
        s_table.add_column("Intrinsic Value (₹)", justify="right", style="bold white")
        s_table.add_column("Current Price (₹)", justify="right")
        s_table.add_column("Margin of Safety", justify="right")

        for name, r in scen_results.items():
            col = "green" if r.bridge.margin_of_safety_pct >= 0 else "red"
            s_table.add_row(
                name,
                format_percentage(r.wacc),
                format_percentage(r.terminal_growth_rate),
                f"₹{r.bridge.intrinsic_value_per_share:,.2f}",
                f"₹{r.bridge.current_share_price:,.2f}",
                f"[{col}]{r.bridge.margin_of_safety_pct * 100:+.1f}%[/{col}]",
            )
        console.print(s_table)
        return

    # Standard DCF run
    asm = ForecastAssumptions(
        forecast_years=years,
        revenue_growth_rates=[growth] * years if growth is not None else None,
        wacc_override=wacc,
        terminal_growth_rate=terminal_growth,
    )
    res = calculate_dcf(profile, forecast_years=years, assumptions=asm)

    # 1. Forecast Schedule Table
    f_table = Table(title=f"{years}-Year Explicit Financial Forecasts: {profile.symbol}", header_style="bold cyan")
    f_table.add_column("Year", justify="center")
    f_table.add_column("Rev Growth", justify="right")
    f_table.add_column("Revenue (₹ Cr)", justify="right")
    f_table.add_column("EBIT Margin", justify="right")
    f_table.add_column("NOPAT (₹ Cr)", justify="right")
    f_table.add_column("CapEx (₹ Cr)", justify="right")
    f_table.add_column("ΔNWC (₹ Cr)", justify="right")
    f_table.add_column("FCFF (₹ Cr)", justify="right", style="bold white")
    f_table.add_column("PV FCFF (₹ Cr)", justify="right")

    for f in res.annual_forecasts:
        f_table.add_row(
            f"Yr {f.year}",
            f"{f.revenue_growth * 100:.1f}%",
            f"₹{f.revenue / CRORE_DIVISOR:,.1f}",
            f"{f.operating_margin * 100:.1f}%",
            f"₹{f.nopat / CRORE_DIVISOR:,.1f}",
            f"₹{f.capex / CRORE_DIVISOR:,.1f}",
            f"₹{f.change_in_nwc / CRORE_DIVISOR:,.1f}",
            f"₹{f.fcff / CRORE_DIVISOR:,.1f}",
            f"₹{f.pv_fcff / CRORE_DIVISOR:,.1f}",
        )
    console.print(f_table)

    # 2. Enterprise-to-Equity Bridge
    b = res.bridge
    b_table = Table(title=f"Enterprise Value to Equity Value Bridge: {profile.company_name} ({profile.symbol})", header_style="bold green")
    b_table.add_column("Valuation Bridge Component", style="cyan")
    b_table.add_column("Value (₹ Cr / ₹ Per Share)", style="bold white", justify="right")

    b_table.add_row("Present Value of Explicit Forecasts", format_crores(b.sum_pv_fcff))
    b_table.add_row(f"Present Value of Terminal Value ({b.terminal_value_pct_of_ev * 100:.1f}% of EV)", format_crores(b.pv_terminal_value))
    b_table.add_row("Implied Enterprise Value", format_crores(b.enterprise_value))
    b_table.add_row("(+) Total Liquid Cash & Short-Term Investments", format_crores(b.total_liquid_assets))
    b_table.add_row("(-) Total Interest-Bearing Debt", format_crores(-b.total_interest_bearing_debt))
    b_table.add_row("(-) Capital Lease Obligations", format_crores(-b.lease_liabilities))
    b_table.add_row("(-) Non-Controlling (Minority) Interest", format_crores(-b.minority_interest))
    b_table.add_row("Implied Equity Value", format_crores(b.implied_equity_value))
    b_table.add_row("Diluted Shares Outstanding", f"{b.diluted_shares / 10_000_000:,.2f} Cr shares")
    b_table.add_row("---", "---")
    b_table.add_row("[bold]Intrinsic Value per Share[/bold]", f"[bold yellow]₹{b.intrinsic_value_per_share:,.2f}[/bold yellow]")
    b_table.add_row("Current Share Price", f"₹{b.current_share_price:,.2f}")
    upside_color = "green" if b.margin_of_safety_pct >= 0 else "red"
    b_table.add_row("Margin of Safety (Implied Upside/Downside)", f"[{upside_color}]{b.margin_of_safety_pct * 100:+.1f}%[/{upside_color}]")
    console.print(b_table)

    # 3. Sensitivity Matrix
    sens_table = Table(title="2D Sensitivity Matrix: Intrinsic Value per Share (₹)", header_style="bold magenta")
    sens_table.add_column("WACC \\ g", style="bold cyan")
    for col in res.sensitivity_table.columns:
        sens_table.add_column(str(col), justify="right")

    for idx, row in res.sensitivity_table.iterrows():
        row_vals = [f"₹{v:,.2f}" if str(v) != "nan" else "N/A" for v in row]
        sens_table.add_row(str(idx), *row_vals)

    console.print(sens_table)

    if res.warnings:
        for w in res.warnings:
            console.print(f"[bold yellow]⚠️ Note:[/bold yellow] [dim]{w}[/dim]")


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
        dcf_res = calculate_dcf(profile) if not profile.is_financial_institution() else None
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
            if profile.is_financial_institution():
                console.print("[bold red]Cannot render industrial DCF chart for financial institution.[/bold red]")
                return
            dcf_res = calculate_dcf(profile)
            plot_dcf_waterfall(dcf_res, save_path=output)

    console.print(f"[bold green]✅ Chart successfully saved to:[/bold green] [bold cyan]{output}[/bold cyan]")


if __name__ == "__main__":
    app()
