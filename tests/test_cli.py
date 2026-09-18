"""CLI smoke tests."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from utterscope import __version__
from utterscope.cli.app import app

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_analyze_stub_succeeds(tmp_path: Path) -> None:
    audio = tmp_path / "lesson.mp3"
    audio.write_bytes(b"fake")
    output_dir = tmp_path / "out"

    result = runner.invoke(
        app,
        [
            "analyze",
            str(audio),
            "--model",
            "tiny",
            "--no-llm",
            "--output",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0
    assert "Pipeline stub" in result.output
    assert "transcript.json" in result.output
    assert output_dir.is_dir()


def test_analyze_missing_file_fails() -> None:
    result = runner.invoke(app, ["analyze", "does-not-exist.mp3"])
    assert result.exit_code != 0
