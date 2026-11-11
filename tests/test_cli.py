"""Unit tests for indiequity CLI flags and commands."""

from typer.testing import CliRunner
from indiequity.cli import app
import indiequity

runner = CliRunner()


def test_cli_version_flag():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert indiequity.__version__ in result.output


def test_cli_short_version_flag():
    result = runner.invoke(app, ["-v"])
    assert result.exit_code == 0
    assert indiequity.__version__ in result.output


def test_cli_help_flag():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Indian Equity Research Toolkit" in result.output
