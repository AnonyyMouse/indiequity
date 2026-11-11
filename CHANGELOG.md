# Changelog

All notable changes to `indiequity` will be documented in this file.

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
