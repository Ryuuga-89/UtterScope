"""CLI smoke tests."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from utterscope import __version__
from utterscope.cli.app import app, format_duration

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_format_duration() -> None:
    assert format_duration(74) == "1m 14s"
    assert format_duration(3661) == "1h 1m 01s"


def test_analyze_prepares_audio(sample_audio: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "out"

    result = runner.invoke(
        app,
        [
            "analyze",
            str(sample_audio),
            "--model",
            "tiny",
            "--no-llm",
            "--output",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0
    assert "prepare audio" in result.output
    assert "Pipeline stub" in result.output
    assert "transcript.json" in result.output
    assert (output_dir / ".utterscope" / "prepared.wav").is_file()


def test_analyze_missing_file_fails() -> None:
    result = runner.invoke(app, ["analyze", "does-not-exist.mp3"])
    assert result.exit_code != 0


def test_analyze_invalid_audio_fails(tmp_path: Path) -> None:
    audio = tmp_path / "broken.mp3"
    audio.write_bytes(b"not-audio")

    result = runner.invoke(
        app,
        ["analyze", str(audio), "--no-llm", "--output", str(tmp_path / "out")],
    )

    assert result.exit_code != 0
    assert "Error:" in result.output
