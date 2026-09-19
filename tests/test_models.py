"""Tests for public pipeline models."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from utterscope.models import (
    ANALYSIS_SCHEMA_VERSION,
    DEFAULT_LONG_PAUSE_THRESHOLD_SECONDS,
    SCHEMA_VERSION,
    AnalysisDocument,
    AnalyzeRequest,
    AnalyzeResult,
    Segment,
    SpeakerRole,
    SpeakingMetrics,
    Transcript,
    TranscriptDocument,
)


def test_segment_rejects_inverted_time_range() -> None:
    with pytest.raises(ValidationError):
        Segment(start=2.0, end=1.0, text="hello")


def test_segment_speaker_defaults_to_none() -> None:
    segment = Segment(start=0.0, end=1.0, text="Hello")
    assert segment.speaker is None


def test_transcript_document_round_trip() -> None:
    document = TranscriptDocument(
        source_audio="lesson.mp3",
        model="large-v3-turbo",
        transcript=Transcript(
            language="en",
            segments=[
                Segment(
                    start=0.0,
                    end=1.5,
                    text="Hello.",
                    speaker="SPEAKER_00",
                ),
                Segment(
                    start=1.5,
                    end=3.0,
                    text="How are you?",
                    speaker="SPEAKER_01",
                ),
            ],
            full_text="Hello. How are you?",
        ),
    )

    restored = TranscriptDocument.model_validate_json(document.model_dump_json())

    assert restored.schema_version == SCHEMA_VERSION
    assert restored.source_audio == "lesson.mp3"
    assert restored.transcript.full_text == "Hello. How are you?"
    assert len(restored.transcript.segments) == 2
    assert restored.transcript.segments[0].speaker == "SPEAKER_00"


def test_analysis_document_round_trip() -> None:
    document = AnalysisDocument(
        source_audio="lesson.mp3",
        model="large-v3-turbo",
        learner_speaker="SPEAKER_01",
        speakers=[
            SpeakerRole(speaker_id="SPEAKER_00", role="other"),
            SpeakerRole(speaker_id="SPEAKER_01", role="student"),
        ],
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

    restored = AnalysisDocument.model_validate_json(document.model_dump_json())

    assert restored.schema_version == ANALYSIS_SCHEMA_VERSION
    assert restored.learner_speaker == "SPEAKER_01"
    assert restored.long_pause_threshold_seconds == (
        DEFAULT_LONG_PAUSE_THRESHOLD_SECONDS
    )
    assert restored.metrics.speaking_ratio == 0.4


def test_analyze_result_can_include_analysis() -> None:
    request = AnalyzeRequest(
        audio_path=Path("lesson.mp3"),
        model="large-v3-turbo",
        output_dir=Path("results"),
        llm=False,
        learner_speaker="SPEAKER_01",
        long_pause_threshold_seconds=1.5,
    )
    transcript_document = TranscriptDocument(
        source_audio=request.audio_path.name,
        model=request.model,
        transcript=Transcript(full_text="Hi.", segments=[]),
    )
    analysis_document = AnalysisDocument(
        source_audio=request.audio_path.name,
        model=request.model,
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
        long_pause_threshold_seconds=request.long_pause_threshold_seconds,
    )
    result = AnalyzeResult(
        document=transcript_document,
        transcript_path=Path("results/transcript.json"),
        duration_seconds=12.5,
        analysis_document=analysis_document,
        analysis_path=Path("results/analysis.json"),
    )

    assert result.analysis_document is not None
    assert result.analysis_path is not None
    assert request.learner_speaker == "SPEAKER_01"
    assert request.long_pause_threshold_seconds == 1.5
