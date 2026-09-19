"""CLI smoke tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from utterscope import __version__
from utterscope.cli.app import app
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
                segments=[
                    Segment(
                        start=0.0,
                        end=1.0,
                        text="Hello",
                        speaker="SPEAKER_01",
                    )
                ],
                full_text="Hello",
            ),
        ),
        transcript_path=output_dir / "transcript.json",
        duration_seconds=1.0,
        learner_speaker="SPEAKER_01",
    )

    def fake_run(
        request,
        *,
        asr=None,
        vad=None,
        diarization=None,
        learner_selector=None,
        progress=None,
        quiet=True,
    ):
        assert progress is not None
        assert learner_selector is not None
        assert quiet is True
        progress.begin("prepare audio")
        progress.end("prepare audio", "1.0s")
        progress.begin("detect speech")
        progress.end("detect speech", "2 intervals")
        progress.begin("transcribe")
        progress.end("transcribe", "1 segments")
        progress.begin("identify speakers")
        progress.end("identify speakers", "2 speakers")
        progress.begin("write transcript")
        progress.end("write transcript")
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
                "--learner",
                "SPEAKER_01",
                "--output",
                str(output_dir),
            ],
        )

    assert result.exit_code == 0
    assert "prepare audio" in result.output
    assert "✔" in result.output
    assert "detect speech" in result.output
    assert "identify speakers" in result.output
    assert "transcribe" in result.output
    assert "Learner → SPEAKER_01" in result.output
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
