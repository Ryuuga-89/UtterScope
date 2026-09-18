"""Tests for JSON report writing."""

from __future__ import annotations

from pathlib import Path

from utterscope.models import Segment, Transcript, TranscriptDocument
from utterscope.report import write_transcript_document


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
