"""Tests for JSON report writing."""

from __future__ import annotations

from pathlib import Path

from utterscope.models import (
    AnalysisDocument,
    Segment,
    SpeakerRole,
    SpeakingMetrics,
    Transcript,
    TranscriptDocument,
)
from utterscope.report import write_analysis_document, write_transcript_document


def test_write_transcript_document(tmp_path: Path) -> None:
    path = tmp_path / "transcript.json"
    document = TranscriptDocument(
        source_audio="lesson.mp3",
        model="tiny",
        transcript=Transcript(
            language="en",
            segments=[Segment(start=0.0, end=1.0, text="Hi")],
            full_text="Hi",
        ),
    )

    written = write_transcript_document(document, path)
    restored = TranscriptDocument.model_validate_json(written.read_text())

    assert written == path
    assert restored.transcript.full_text == "Hi"


def test_write_analysis_document(tmp_path: Path) -> None:
    path = tmp_path / "analysis.json"
    document = AnalysisDocument(
        source_audio="lesson.mp3",
        model="tiny",
        learner_speaker="SPEAKER_01",
        speakers=[SpeakerRole(speaker_id="SPEAKER_01", role="student")],
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
    )

    written = write_analysis_document(document, path)
    restored = AnalysisDocument.model_validate_json(written.read_text())

    assert written == path
    assert restored.metrics.wpm == 60.0
