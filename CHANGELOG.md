# Changelog

All notable changes to `indiequity` will be documented in this file.

## [0.2.0] - Valuation Engine Overhaul & Financial Methodology Repair
### Fixed & Corrected
- **FCFF Formulation**: Replaced mechanical proxy with corporate finance identity: `FCFF = EBIT * (1 - t) + D&A - CapEx - ΔOperating_NWC`.
- **Negative FCFF Preservation**: Retained genuine negative cash flows without replacing them with arbitrary positive numbers.
- **Explicit Annual Projections**: Replaced simplistic 12% single-growth compounding with line-by-line forecasts (Revenue, Margins, Taxes, CapEx, ΔNWC, Reinvestment Rate, and ROIC) for 5-year and 10-year horizons.
- **WACC Estimation & Capital Structure**: Calculated WACC using market capitalization and explicit interest-bearing debt; removed arbitrary clamping; added validation error when terminal growth >= WACC.
- **Sustainable Terminal Value**: Linked perpetual Gordon growth to sustainable terminal reinvestment (`g / Terminal_ROIC`), preventing inflated terminal values.
- **Enterprise-to-Equity Bridge**: Reconciled EV to Equity Value by accounting for liquid cash, interest-bearing debt, lease obligations, and minority interests.
- **Data Quality & Period Ordering**: Chronologically sorted balance sheet and P&L statements (newest to oldest); eliminated silent replacement of missing items with zero.

### Added
- **Financial Institutions Routing**: Blocked industrial FCFF/WACC DCF on Banks/NBFCs; introduced Residual Income Model (RIM) and Justified P/B valuation.
- **Reverse DCF**: Implemented market-implied growth solver (`reverse_dcf()`) to quantify growth expectations embedded in current stock price.
- **Scenario Analysis**: Added Bear, Base, and Bull valuation cases with explicit deviations in growth, margins, and cost of capital.
- **Excel Export Overhaul**: Added explicit forecast schedules and comprehensive EV-to-equity reconciliation tabs in generated `.xlsx` workbooks.
- **Rigorous Test Suite**: Expanded unit tests to 24 tests covering the worked ₹100 Cr FCFF example, missing data rejection, monotonic sensitivity, and bank routing.

## [0.1.0] - Initial Release
- Fundamental public financial data ingestion for NSE and BSE equities.
- Financial ratios: ROE, ROCE, D/E, Interest Coverage, and FCF Yield in ₹ Crores.
- 3-Stage and 5-Stage DuPont ROE decomposition.
- Stock screener with configurable filters and investment presets.
- 5-Year FCFF DCF model with a 2D sensitivity table (WACC vs. g).
- Matplotlib chart generation (DuPont bar chart, DCF waterfall).
- Multi-tab Excel export (`.xlsx`) via openpyxl.
- Interactive terminal CLI using Rich and Typer.
- Offline unit tests with mock small-cap financial fixtures.
