"""SQLite-backed lesson history index."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from utterscope.config import resolve_db_path
from utterscope.models import AnalyzeResult

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    run_dir TEXT NOT NULL UNIQUE,
    run_name TEXT NOT NULL,
    source_audio TEXT NOT NULL,
    duration_seconds REAL,
    asr_model TEXT NOT NULL,
    learner_speaker TEXT,
    long_pause_threshold_seconds REAL,
    speaking_time_seconds REAL,
    speaking_ratio REAL,
    wpm REAL,
    turn_count INTEGER,
    average_turn_seconds REAL,
    pause_count INTEGER,
    long_pause_count INTEGER,
    filler_count INTEGER,
    llm_enabled INTEGER NOT NULL,
    llm_provider TEXT,
    llm_model TEXT,
    issue_count INTEGER NOT NULL DEFAULT 0,
    pattern_count INTEGER NOT NULL DEFAULT 0
);
"""


@dataclass(frozen=True)
class HistoryRun:
    """One indexed analyze run for history listing."""

    created_at: str
    run_dir: str
    run_name: str
    source_audio: str
    duration_seconds: float | None
    asr_model: str
    learner_speaker: str | None
    speaking_time_seconds: float | None
    speaking_ratio: float | None
    wpm: float | None
    turn_count: int | None
    filler_count: int | None
    llm_enabled: bool
    issue_count: int
    pattern_count: int


def connect(db_path: Path | None = None) -> sqlite3.Connection:
    """Open the history database, creating the parent directory and schema."""
    path = resolve_db_path() if db_path is None else db_path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(_SCHEMA)
    return connection


def record_run(
    result: AnalyzeResult,
    *,
    db_path: Path | None = None,
    created_at: datetime | None = None,
) -> Path:
    """Insert or update the history row for ``result.run_dir``.

    Returns the resolved database path.
    """
    if result.run_dir is None:
        msg = "run_dir is required to record history"
        raise ValueError(msg)
    if result.analysis_document is None:
        msg = "analysis_document is required to record history"
        raise ValueError(msg)

    run_dir = result.run_dir.expanduser().resolve()
    analysis = result.analysis_document
    metrics = analysis.metrics
    feedback = result.feedback_document
    timestamp = created_at or datetime.now().astimezone()

    path = resolve_db_path() if db_path is None else db_path.expanduser().resolve()
    with connect(path) as connection:
        connection.execute(
            """
            INSERT INTO runs (
                created_at,
                run_dir,
                run_name,
                source_audio,
                duration_seconds,
                asr_model,
                learner_speaker,
                long_pause_threshold_seconds,
                speaking_time_seconds,
                speaking_ratio,
                wpm,
                turn_count,
                average_turn_seconds,
                pause_count,
                long_pause_count,
                filler_count,
                llm_enabled,
                llm_provider,
                llm_model,
                issue_count,
                pattern_count
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            ON CONFLICT(run_dir) DO UPDATE SET
                created_at = excluded.created_at,
                run_name = excluded.run_name,
                source_audio = excluded.source_audio,
                duration_seconds = excluded.duration_seconds,
                asr_model = excluded.asr_model,
                learner_speaker = excluded.learner_speaker,
                long_pause_threshold_seconds =
                    excluded.long_pause_threshold_seconds,
                speaking_time_seconds = excluded.speaking_time_seconds,
                speaking_ratio = excluded.speaking_ratio,
                wpm = excluded.wpm,
                turn_count = excluded.turn_count,
                average_turn_seconds = excluded.average_turn_seconds,
                pause_count = excluded.pause_count,
                long_pause_count = excluded.long_pause_count,
                filler_count = excluded.filler_count,
                llm_enabled = excluded.llm_enabled,
                llm_provider = excluded.llm_provider,
                llm_model = excluded.llm_model,
                issue_count = excluded.issue_count,
                pattern_count = excluded.pattern_count
            """,
            (
                timestamp.isoformat(timespec="seconds"),
                str(run_dir),
                run_dir.name,
                analysis.source_audio,
                result.duration_seconds,
                analysis.model,
                result.learner_speaker or analysis.learner_speaker,
                analysis.long_pause_threshold_seconds,
                metrics.speaking_time_seconds,
                metrics.speaking_ratio,
                metrics.wpm,
                metrics.turn_count,
                metrics.average_turn_seconds,
                metrics.pause_count,
                metrics.long_pause_count,
                metrics.filler_count,
                1 if feedback is not None else 0,
                feedback.llm_provider if feedback is not None else None,
                feedback.llm_model if feedback is not None else None,
                len(feedback.issues) if feedback is not None else 0,
                (len(feedback.recurring_patterns) if feedback is not None else 0),
            ),
        )
        connection.commit()
    return path


def list_runs(
    *,
    limit: int = 20,
    db_path: Path | None = None,
) -> list[HistoryRun]:
    """Return recent runs newest-first."""
    if limit < 0:
        msg = "limit must be >= 0"
        raise ValueError(msg)

    path = resolve_db_path() if db_path is None else db_path.expanduser().resolve()
    if not path.is_file():
        return []

    with connect(path) as connection:
        rows = connection.execute(
            """
            SELECT
                created_at,
                run_dir,
                run_name,
                source_audio,
                duration_seconds,
                asr_model,
                learner_speaker,
                speaking_time_seconds,
                speaking_ratio,
                wpm,
                turn_count,
                filler_count,
                llm_enabled,
                issue_count,
                pattern_count
            FROM runs
            ORDER BY datetime(created_at) DESC, id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return [
        HistoryRun(
            created_at=row["created_at"],
            run_dir=row["run_dir"],
            run_name=row["run_name"],
            source_audio=row["source_audio"],
            duration_seconds=row["duration_seconds"],
            asr_model=row["asr_model"],
            learner_speaker=row["learner_speaker"],
            speaking_time_seconds=row["speaking_time_seconds"],
            speaking_ratio=row["speaking_ratio"],
            wpm=row["wpm"],
            turn_count=row["turn_count"],
            filler_count=row["filler_count"],
            llm_enabled=bool(row["llm_enabled"]),
            issue_count=int(row["issue_count"]),
            pattern_count=int(row["pattern_count"]),
        )
        for row in rows
    ]
