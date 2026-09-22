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
    output_dir.mkdir()
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
        run_dir=output_dir,
    )

    def fake_run(
        request,
        *,
        backend=None,
        vad=None,
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
        progress.begin("transcribe + identify speakers")
        progress.end("transcribe + identify speakers", "1 segments, 2 speakers")
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
    assert "record history" in result.output
    assert "Learner → SPEAKER_01" in result.output
    assert "Transcript →" in result.output
    assert "Analysis →" in result.output
    assert "Metrics →" in result.output


def test_history_empty(tmp_path: Path, monkeypatch) -> None:
    from utterscope.config import DB_PATH_KEY

    monkeypatch.setenv(DB_PATH_KEY, str(tmp_path / "empty.sqlite"))
    result = runner.invoke(app, ["history"])
    assert result.exit_code == 0
    assert "履歴はありません" in result.output


def test_history_lists_recorded_run(tmp_path: Path, monkeypatch) -> None:
    from utterscope.config import DB_PATH_KEY
    from utterscope.storage import record_run

    db_path = tmp_path / "listed.sqlite"
    monkeypatch.setenv(DB_PATH_KEY, str(db_path))
    run_dir = tmp_path / "250919-1_lesson"
    run_dir.mkdir()
    result_doc = AnalyzeResult(
        document=TranscriptDocument(
            source_audio="lesson.mp3",
            model="tiny",
            transcript=Transcript(
                language="en",
                segments=[
                    Segment(start=0.0, end=1.0, text="Hi", speaker="SPEAKER_01")
                ],
                full_text="Hi",
            ),
        ),
        transcript_path=run_dir / "transcript.json",
        duration_seconds=1.0,
        learner_speaker="SPEAKER_01",
        analysis_document=AnalysisDocument(
            source_audio="lesson.mp3",
            model="tiny",
            learner_speaker="SPEAKER_01",
            speakers=[SpeakerRole(speaker_id="SPEAKER_01", role="student")],
            metrics=SpeakingMetrics(
                speaking_time_seconds=1.0,
                speaking_ratio=1.0,
                wpm=72.0,
                turn_count=1,
                average_turn_seconds=1.0,
                pause_count=0,
                long_pause_count=0,
                filler_count=2,
            ),
        ),
        analysis_path=run_dir / "analysis.json",
        run_dir=run_dir,
    )
    record_run(result_doc, db_path=db_path)

    result = runner.invoke(app, ["history", "--limit", "5"], env={"COLUMNS": "200"})
    assert result.exit_code == 0
    assert "Lesson history" in result.output
    assert "72" in result.output
    # Rich may wrap long paths; assert via storage API as well.
    from utterscope.storage import list_runs

    runs = list_runs(limit=5, db_path=db_path)
    assert len(runs) == 1
    assert runs[0].run_name == "250919-1_lesson"


def test_analyze_requires_learner(sample_audio: Path, tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "analyze",
            str(sample_audio),
            "--no-llm",
            "--output",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code != 0
    assert "--learner" in result.output


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
            "--learner",
            "SPEAKER_01",
            "--long-pause-threshold",
            "0",
            "--output",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code != 0
    assert "greater than 0" in result.output


def test_analyze_rejects_negative_report_turn_gap(
    sample_audio: Path, tmp_path: Path
) -> None:
    result = runner.invoke(
        app,
        [
            "analyze",
            str(sample_audio),
            "--no-llm",
            "--learner",
            "SPEAKER_01",
            "--report-turn-gap",
            "-1",
            "--output",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code != 0
    assert "greater than or equal to 0" in result.output


def test_analyze_invalid_audio_fails(tmp_path: Path) -> None:
    audio = tmp_path / "broken.mp3"
    audio.write_bytes(b"not-audio")

    result = runner.invoke(
        app,
        [
            "analyze",
            str(audio),
            "--no-llm",
            "--learner",
            "SPEAKER_01",
            "--output",
            str(tmp_path / "out"),
        ],
    )

    assert result.exit_code != 0
    assert "Error:" in result.output
