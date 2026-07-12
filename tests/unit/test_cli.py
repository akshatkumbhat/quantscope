\"\"\"Tests for the QuantScope CLI.\"\"\"

from typer.testing import CliRunner

from quantscope.cli import app

runner = CliRunner()


def test_cli_help() -> None:
    result = runner.invoke(app, [\"--help\"])
    assert result.exit_code == 0
    assert \"quantization\" in result.stdout.lower()


def test_version_info() -> None:
    result = runner.invoke(app, [\"version-info\"])
    assert result.exit_code == 0
    assert \"QuantScope\" in result.stdout
