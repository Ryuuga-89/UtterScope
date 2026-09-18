"""Tests for public pipeline models."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from utterscope.models import (
    SCHEMA_VERSION,
    AnalyzeRequest,
    AnalyzeResult,
    Segment,
    Transcript,
    TranscriptDocument,
)


def test_segment_rejects_inverted_time_range() -> None:
    with pytest.raises(ValidationError):
        Segment(start=2.0, end=1.0, text="hello")


def test_transcript_document_round_trip() -> None:
    document = TranscriptDocument(
        source_audio="lesson.mp3",
        model="large-v3-turbo",
        transcript=Transcript(
            language="en",
            segments=[
                Segment(start=0.0, end=1.5, text="Hello."),
                Segment(start=1.5, end=3.0, text="How are you?"),
            ],
            full_text="Hello. How are you?",
        ),
    )

    restored = TranscriptDocument.model_validate_json(document.model_dump_json())

    assert restored.schema_version == SCHEMA_VERSION
    assert restored.source_audio == "lesson.mp3"
    assert restored.transcript.full_text == "Hello. How are you?"
    assert len(restored.transcript.segments) == 2


def test_analyze_result_keeps_transcript_under_document() -> None:
    request = AnalyzeRequest(
        audio_path=Path("lesson.mp3"),
        model="large-v3-turbo",
        output_dir=Path("results"),
        llm=False,
    )
    document = TranscriptDocument(
        source_audio=request.audio_path.name,
        model=request.model,
        transcript=Transcript(full_text="Hi.", segments=[]),
    )
    result = AnalyzeResult(
        document=document,
        transcript_path=Path("results/transcript.json"),
        duration_seconds=12.5,
    )

    assert result.document.transcript.full_text == "Hi."
    assert request.llm is False
