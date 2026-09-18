"""CLI smoke tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from utterscope import __version__
from utterscope.cli.app import app, format_duration
from utterscope.models import (
    AnalyzeResult,
    Segment,
    Transcript,
    TranscriptDocument,
)

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_format_duration() -> None:
    assert format_duration(74) == "1m 14s"
    assert format_duration(3661) == "1h 1m 01s"


def test_analyze_reports_progress(
    sample_audio: Path, tmp_path: Path
) -> None:
    output_dir = tmp_path / "out"
    fake_result = AnalyzeResult(
        document=TranscriptDocument(
            source_audio=sample_audio.name,
            model="tiny",
            transcript=Transcript(
                language="en",
                segments=[Segment(start=0.0, end=1.0, text="Hello")],
                full_text="Hello",
            ),
        ),
        transcript_path=output_dir / "transcript.json",
        duration_seconds=1.0,
    )

    def fake_run(request, *, asr=None, progress=None):
        assert progress is not None
        progress.on_prepare_audio(1.0)
        progress.on_transcribe(1)
        progress.on_write_transcript(fake_result.transcript_path)
        return fake_result

    with patch("utterscope.cli.app.run_pipeline", side_effect=fake_run):
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
    assert "transcribe" in result.output
    assert "Transcript →" in result.output


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
