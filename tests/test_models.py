"""Tests for public pipeline models."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from utterscope.models import (
    ANALYSIS_SCHEMA_VERSION,
    DEFAULT_FEEDBACK_CONTEXT_TURNS,
    DEFAULT_LLM_MODEL,
    DEFAULT_LONG_PAUSE_THRESHOLD_SECONDS,
    FEEDBACK_SCHEMA_VERSION,
    SCHEMA_VERSION,
    AnalysisDocument,
    AnalyzeRequest,
    AnalyzeResult,
    FeedbackDocument,
    FeedbackIssue,
    RecurringPattern,
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


def test_feedback_document_round_trip() -> None:
    document = FeedbackDocument(
        source_audio="lesson.mp3",
        asr_model="large-v3-turbo",
        llm_model=DEFAULT_LLM_MODEL,
        learner_speaker="SPEAKER_01",
        summary="Overall the learner communicates clearly but verb tense slips.",
        recurring_patterns=[
            RecurringPattern(
                category="grammar",
                label="past-tense omission",
                count=3,
                examples=["I go yesterday", "I eat lunch"],
                message="Past events are often described with present verbs.",
            )
        ],
        issues=[
            FeedbackIssue(
                category="grammar",
                severity="high",
                scope="phrase",
                turn_index=2,
                start=10.0,
                end=12.5,
                excerpt="I go yesterday",
                message="Use past tense for a completed past event.",
                suggestion="I went yesterday",
            ),
            FeedbackIssue(
                category="naturalness",
                scope="turn",
                turn_index=4,
                start=20.0,
                end=23.0,
                excerpt="Yes I am agree with you",
                message="This agreement phrase is unnatural.",
                suggestion="Yes, I agree with you",
            ),
        ],
        context_turns=DEFAULT_FEEDBACK_CONTEXT_TURNS,
    )

    restored = FeedbackDocument.model_validate_json(document.model_dump_json())

    assert restored.schema_version == FEEDBACK_SCHEMA_VERSION
    assert restored.summary.startswith("Overall")
    assert restored.llm_provider == "gemini"
    assert restored.context_turns == 1
    assert len(restored.issues) == 2
    assert restored.issues[0].scope == "phrase"
    assert restored.recurring_patterns[0].count == 3


def test_feedback_issue_rejects_inverted_time_range() -> None:
    with pytest.raises(ValidationError):
        FeedbackIssue(
            category="vocabulary",
            turn_index=0,
            start=2.0,
            end=1.0,
            excerpt="hello",
            message="test",
        )


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
        feedback_document=FeedbackDocument(
            source_audio=request.audio_path.name,
            asr_model=request.model,
            learner_speaker="SPEAKER_01",
            summary="Short overview of the learner's performance.",
        ),
        feedback_path=Path("results/feedback.json"),
    )

    assert result.analysis_document is not None
    assert result.analysis_path is not None
    assert result.feedback_document is not None
    assert result.feedback_path is not None
    assert request.learner_speaker == "SPEAKER_01"
    assert request.long_pause_threshold_seconds == 1.5
    assert request.llm_model == DEFAULT_LLM_MODEL
    assert request.feedback_context_turns == DEFAULT_FEEDBACK_CONTEXT_TURNS
