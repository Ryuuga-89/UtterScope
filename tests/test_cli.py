"""CLI smoke tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from utterscope import __version__
from utterscope.cli.app import app
from utterscope.models import (
    AnalysisDocument,
    AnalyzeResult,
    Segment,
    SpeakerRole,
    SpeakingMetrics,
    Transcript,
    TranscriptDocument,
)

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_analyze_reports_progress(sample_audio: Path, tmp_path: Path) -> None:
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
        analysis_path=output_dir / "analysis.json",
        analysis_document=AnalysisDocument(
            source_audio=sample_audio.name,
            model="tiny",
            learner_speaker="SPEAKER_01",
            speakers=[
                SpeakerRole(speaker_id="SPEAKER_01", role="student"),
            ],
            metrics=SpeakingMetrics(
                speaking_time_seconds=1.0,
                speaking_ratio=1.0,
                wpm=60.0,
                turn_count=1,
                average_turn_seconds=1.0,
                pause_count=0,
                long_pause_count=0,
                filler_count=0,
            ),
        ),
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
        progress.begin("analyze learner speech")
        progress.end("analyze learner speech", "60 wpm")
        progress.begin("write results")
        progress.end("write results")
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
    assert "analyze learner speech" in result.output
    assert "Learner → SPEAKER_01" in result.output
    assert "Transcript →" in result.output
    assert "Analysis →" in result.output
    assert "Metrics →" in result.output


def test_analyze_rejects_llm_without_gemini_key(
    sample_audio: Path, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with patch("utterscope.cli.app.read_env_value", return_value=None):
        result = runner.invoke(
            app,
            [
                "analyze",
                str(sample_audio),
                "--llm",
                "--learner",
                "SPEAKER_01",
                "--output",
                str(tmp_path / "out"),
            ],
        )
    assert result.exit_code != 0
    assert "GEMINI_API_KEY" in result.output


def test_analyze_rejects_non_positive_long_pause_threshold(
    sample_audio: Path, tmp_path: Path
) -> None:
    result = runner.invoke(
        app,
        [
            "analyze",
            str(sample_audio),
            "--no-llm",
            "--long-pause-threshold",
            "0",
            "--output",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code != 0
    assert "greater than 0" in result.output


def test_analyze_invalid_audio_fails(tmp_path: Path) -> None:
    audio = tmp_path / "broken.mp3"
    audio.write_bytes(b"not-audio")

    result = runner.invoke(
        app,
        ["analyze", str(audio), "--no-llm", "--output", str(tmp_path / "out")],
    )

    assert result.exit_code != 0
    assert "Error:" in result.output
