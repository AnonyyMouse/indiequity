"""
Unit tests for indiequity charts and export modules.
"""

import tempfile
from pathlib import Path
from indiequity.charts import plot_dcf_waterfall, plot_dupont_5stage
from indiequity.data import FinancialProfile
from indiequity.dcf import calculate_dcf
from indiequity.dupont import decompose_dupont_3stage, decompose_dupont_5stage
from indiequity.export import export_to_csv, export_to_excel
from indiequity.ratios import calculate_ratios


def test_charts_generation(mock_indian_smallcap: FinancialProfile):
    """Verify matplotlib chart generation without error."""
    d5 = decompose_dupont_5stage(mock_indian_smallcap)
    fig_dupont = plot_dupont_5stage(d5)
    assert fig_dupont is not None

    dcf_res = calculate_dcf(mock_indian_smallcap)
    fig_dcf = plot_dcf_waterfall(dcf_res)
    assert fig_dcf is not None


def test_excel_export(mock_indian_smallcap: FinancialProfile):
    """Verify multi-tab Excel workbook generation."""
    ratios = calculate_ratios(mock_indian_smallcap)
    d3 = decompose_dupont_3stage(mock_indian_smallcap)
    d5 = decompose_dupont_5stage(mock_indian_smallcap)
    dcf_res = calculate_dcf(mock_indian_smallcap)

    with tempfile.TemporaryDirectory() as tmpdir:
        xlsx_path = Path(tmpdir) / "report.xlsx"
        created_path = export_to_excel(
            mock_indian_smallcap,
            ratios,
            d3,
            d5,
            dcf_res,
            str(xlsx_path),
        )
        assert Path(created_path).exists()
        assert Path(created_path).stat().st_size > 1000
