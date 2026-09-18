"""Smoke tests for the project skeleton."""

from typer.testing import CliRunner

from utterscope import __version__
from utterscope.cli.app import app

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout
