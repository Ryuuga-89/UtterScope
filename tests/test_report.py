"""Tests for JSON / Markdown / HTML report writing."""

from __future__ import annotations

from pathlib import Path

from utterscope.models import (
    AnalysisDocument,
    FeedbackDocument,
    FeedbackIssue,
    RecurringPattern,
    Segment,
    SpeakerRole,
    SpeakingMetrics,
    Transcript,
    TranscriptDocument,
)
from utterscope.report import (
    write_analysis_document,
    write_reports,
    write_transcript_document,
)
from utterscope.report.generate import REPORT_HTML_FILENAME, REPORT_MD_FILENAME
from utterscope.report.labels import format_timestamp


def _sample_documents() -> tuple[
    TranscriptDocument, AnalysisDocument, FeedbackDocument
]:
    transcript = TranscriptDocument(
        source_audio="lesson.mp3",
        model="tiny",
        transcript=Transcript(
            language="en",
            segments=[
                Segment(
                    start=0.0,
                    end=1.5,
                    text="Hello there.",
                    speaker="SPEAKER_00",
                ),
                Segment(
                    start=1.5,
                    end=3.0,
                    text="I go yesterday.",
                    speaker="SPEAKER_01",
                ),
            ],
            full_text="Hello there. I go yesterday.",
        ),
    )
    analysis = AnalysisDocument(
        source_audio="lesson.mp3",
        model="tiny",
        learner_speaker="SPEAKER_01",
        speakers=[
            SpeakerRole(speaker_id="SPEAKER_00", role="other"),
            SpeakerRole(speaker_id="SPEAKER_01", role="student"),
        ],
        metrics=SpeakingMetrics(
            speaking_time_seconds=1.5,
            speaking_ratio=0.5,
            wpm=80.0,
            turn_count=1,
            average_turn_seconds=1.5,
            pause_count=1,
            long_pause_count=0,
            filler_count=0,
        ),
    )
    feedback = FeedbackDocument(
        source_audio="lesson.mp3",
        asr_model="tiny",
        llm_provider="gemini",
        llm_model="gemini-3.8-flash",
        learner_speaker="SPEAKER_01",
        summary="全体として意思疎通はできています。",
        recurring_patterns=[
            RecurringPattern(
                category="grammar",
                label="過去形の抜け",
                count=1,
                examples=["I go yesterday"],
                message="過去の出来事を現在形で述べがちです。",
            )
        ],
        issues=[
            FeedbackIssue(
                category="grammar",
                severity="high",
                scope="phrase",
                turn_index=1,
                start=1.5,
                end=3.0,
                excerpt="I go yesterday",
                message="過去の出来事には過去形を使いましょう。",
                suggestion="I went yesterday",
            )
        ],
    )
    return transcript, analysis, feedback


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


def test_format_timestamp() -> None:
    assert format_timestamp(65) == "1:05"
    assert format_timestamp(3661) == "1:01:01"


def test_write_reports_with_feedback(tmp_path: Path) -> None:
    transcript, analysis, feedback = _sample_documents()
    md_path, html_path = write_reports(
        tmp_path,
        transcript_document=transcript,
        analysis_document=analysis,
        feedback_document=feedback,
        duration_seconds=3.0,
        audio_filename="lesson.mp3",
    )

    assert md_path == tmp_path / REPORT_MD_FILENAME
    assert html_path == tmp_path / REPORT_HTML_FILENAME
    md = md_path.read_text(encoding="utf-8")
    html = html_path.read_text(encoding="utf-8")

    assert "# 分析レポート" in md
    assert "ターン 1" in md
    assert "I go yesterday" in md
    assert "過去の出来事には過去形を使いましょう" in md
    assert "AI向け" not in md

    assert "分析レポート" in html
    assert 'src="lesson.mp3"' in html
    assert "I go yesterday" in html
    assert "指摘パネル" in html
    assert "cdn." not in html.lower()


def test_write_reports_without_feedback(tmp_path: Path) -> None:
    transcript, analysis, _ = _sample_documents()
    md_path, html_path = write_reports(
        tmp_path,
        transcript_document=transcript,
        analysis_document=analysis,
        feedback_document=None,
        audio_filename="lesson.mp3",
    )

    md = md_path.read_text(encoding="utf-8")
    html = html_path.read_text(encoding="utf-8")
    assert "LLM フィードバックは実行されていません" in md
    assert "LLM フィードバックは実行されていません" in html
    assert "ターン 0" in md
