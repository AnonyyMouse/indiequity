# indiequity: Indian Equity Research Toolkit

A Python library designed for small-cap and mid-cap equity research on Indian markets (NSE & BSE).

## Key Features
- **Public Data Ingestion**: Seamless financial statement retrieval for NSE and BSE tickers with local SQLite caching.
- **Fundamental Ratios**: Return on Capital Employed (ROCE), Return on Equity (ROE), Debt-to-Equity, Interest Coverage, and FCF Yield.
- **DuPont Analysis**: 3-Stage and 5-Stage Return on Equity decomposition.
- **Stock Screener**: Configurable screening filters and built-in strategy presets (`Quality Smallcap`, `Deep Value`, `High Compounder`).
- **5-Year FCFF DCF Valuation**: Enterprise value and intrinsic equity value per share with a 2D sensitivity grid (WACC vs. Terminal Growth Rate).
- **Export & Visualization**: Publication-grade charts and multi-tab formatted Excel workbooks (`.xlsx`).
- **Command Line & Notebook Interfaces**: Clean CLI powered by Rich and Typer, plus interactive Jupyter tutorial notebooks.

## Installation

```bash
pip install indiequity
```

Or install in development mode from source:
```bash
git clone https://github.com/AnonyyMouse/indiequity.git
cd indiequity
pip install -e .
```

## Quickstart

### Python API
```python
import indiequity as ie

# 1. Fetch public financial profile
profile = ie.fetch_company_data("ITC")

# 2. Compute key financial ratios
ratios = ie.calculate_ratios(profile)
print(f"Company:   {profile.company_name}")
print(f"Mcap:      {ie.format_crores(ratios.market_cap)}")
print(f"ROCE:      {ie.format_percentage(ratios.roce)}")
print(f"ROE:       {ie.format_percentage(ratios.roe)}")

# 3. 5-Stage DuPont Decomposition
dupont = ie.decompose_dupont_5stage(profile)
print(f"Operating Margin: {ie.format_percentage(dupont.operating_margin)}")
print(f"Tax Burden:       {dupont.tax_burden:.2f}x")
print(f"Interest Burden:  {dupont.interest_burden:.2f}x")

# 4. DCF Valuation & Sensitivity Analysis
dcf = ie.calculate_dcf(profile, forecast_years=5, terminal_growth_rate=0.05)
print(f"Intrinsic Value: ₹{dcf.intrinsic_value_per_share:,.2f} per share")
print(dcf.sensitivity_table)
```

### Command Line Interface (CLI)
```bash
# View company ratios
indiequity ratios ITC

# View 5-stage DuPont decomposition
indiequity dupont ITC --stages 5

# Run 5-Year DCF with sensitivity table
indiequity dcf ITC

# Run stock screener
indiequity screen --universe smallcap --preset quality-smallcap

# Export complete research report to Excel
indiequity export ITC --output itc_report.xlsx
```

## Known Limitations & Disclaimer
- **Data Source**: Fundamental statements are sourced from public market feeds via `yfinance`. Corporate restructurings or unaudited quarterly results may occasionally require cross-checking with official BSE/NSE filings.
- **Educational / Research Use Only**: This toolkit is created by equity research students for educational, academic, and screening purposes. It does not constitute investment advice or SEBI-registered advisory recommendations.

## License
Distributed under the MIT License. See `LICENSE` for details.
