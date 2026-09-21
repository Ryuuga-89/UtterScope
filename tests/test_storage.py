"""Tests for SQLite lesson history storage."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from utterscope.config import DB_PATH_KEY, resolve_db_path
from utterscope.models import (
    AnalysisDocument,
    AnalyzeResult,
    FeedbackDocument,
    FeedbackIssue,
    Segment,
    SpeakerRole,
    SpeakingMetrics,
    Transcript,
    TranscriptDocument,
)
from utterscope.storage import list_runs, record_run


def _result(tmp_path: Path, *, name: str = "250919-1_lesson") -> AnalyzeResult:
    run_dir = tmp_path / name
    run_dir.mkdir(parents=True, exist_ok=True)
    transcript = TranscriptDocument(
        source_audio="lesson.mp3",
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
    )
    analysis = AnalysisDocument(
        source_audio="lesson.mp3",
        model="tiny",
        learner_speaker="SPEAKER_01",
        speakers=[SpeakerRole(speaker_id="SPEAKER_01", role="student")],
        metrics=SpeakingMetrics(
            speaking_time_seconds=10.0,
            speaking_ratio=0.4,
            wpm=100.0,
            turn_count=5,
            average_turn_seconds=2.0,
            pause_count=4,
            long_pause_count=1,
            filler_count=3,
        ),
    )
    return AnalyzeResult(
        document=transcript,
        transcript_path=run_dir / "transcript.json",
        duration_seconds=30.0,
        learner_speaker="SPEAKER_01",
        analysis_document=analysis,
        analysis_path=run_dir / "analysis.json",
        run_dir=run_dir,
    )


def test_resolve_db_path_default(monkeypatch) -> None:
    monkeypatch.delenv(DB_PATH_KEY, raising=False)
    path = resolve_db_path()
    assert path.name == "history.sqlite"
    assert path.parent.name == ".utterscope"


def test_resolve_db_path_requires_absolute(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(DB_PATH_KEY, "relative.sqlite")
    with pytest.raises(ValueError, match="absolute"):
        resolve_db_path(tmp_path)


def test_record_and_list_runs(tmp_path: Path) -> None:
    db_path = tmp_path / "history.sqlite"
    result = _result(tmp_path)
    written = record_run(result, db_path=db_path)
    assert written == db_path.resolve()

    runs = list_runs(limit=10, db_path=db_path)
    assert len(runs) == 1
    assert runs[0].run_name == "250919-1_lesson"
    assert runs[0].wpm == 100.0
    assert runs[0].filler_count == 3
    assert runs[0].llm_enabled is False
    assert runs[0].issue_count == 0


def test_record_run_upserts_same_run_dir(tmp_path: Path) -> None:
    db_path = tmp_path / "history.sqlite"
    result = _result(tmp_path)
    record_run(
        result,
        db_path=db_path,
        created_at=datetime(2025, 1, 1, tzinfo=UTC),
    )

    feedback = FeedbackDocument(
        source_audio="lesson.mp3",
        asr_model="tiny",
        learner_speaker="SPEAKER_01",
        summary="全体は良好です。",
        issues=[
            FeedbackIssue(
                category="grammar",
                severity="low",
                turn_index=0,
                start=0.0,
                end=1.0,
                excerpt="Hello",
                message="Minor note.",
            )
        ],
    )
    updated = result.model_copy(
        update={
            "feedback_document": feedback,
            "analysis_document": result.analysis_document.model_copy(
                update={
                    "metrics": result.analysis_document.metrics.model_copy(
                        update={"wpm": 120.0}
                    )
                }
            ),
        }
    )
    record_run(
        updated,
        db_path=db_path,
        created_at=datetime(2025, 1, 2, tzinfo=UTC),
    )

    runs = list_runs(limit=10, db_path=db_path)
    assert len(runs) == 1
    assert runs[0].wpm == 120.0
    assert runs[0].llm_enabled is True
    assert runs[0].issue_count == 1
    assert runs[0].created_at.startswith("2025-01-02")


def test_list_runs_respects_limit(tmp_path: Path) -> None:
    db_path = tmp_path / "history.sqlite"
    base = datetime(2025, 6, 1, tzinfo=UTC)
    for index in range(5):
        result = _result(tmp_path, name=f"250601-{index + 1}_lesson")
        record_run(
            result,
            db_path=db_path,
            created_at=base + timedelta(hours=index),
        )

    runs = list_runs(limit=2, db_path=db_path)
    assert len(runs) == 2
    assert runs[0].run_name == "250601-5_lesson"
    assert runs[1].run_name == "250601-4_lesson"


def test_list_runs_missing_db(tmp_path: Path) -> None:
    assert list_runs(db_path=tmp_path / "missing.sqlite") == []
