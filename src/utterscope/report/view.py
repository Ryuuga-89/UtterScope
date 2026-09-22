"""View-model helpers shared by Markdown and HTML report renderers."""

from __future__ import annotations

from dataclasses import dataclass, field

from utterscope.llm.turns import DialogueTurn, build_dialogue_turns
from utterscope.models import (
    DEFAULT_REPORT_TURN_GAP_SECONDS,
    AnalysisDocument,
    FeedbackDocument,
    FeedbackIssue,
    TranscriptDocument,
)
from utterscope.report.labels import (
    category_label,
    format_timestamp,
    role_label,
    severity_label,
)


@dataclass(frozen=True)
class ReportIssueView:
    """One feedback issue prepared for display."""

    category: str
    category_label: str
    severity: str
    severity_label: str
    scope: str
    turn_index: int
    start: float
    end: float
    start_label: str
    end_label: str
    excerpt: str
    message: str
    suggestion: str | None


@dataclass(frozen=True)
class ReportTurnView:
    """One dialogue turn with attached issues."""

    index: int
    speaker_id: str
    role: str | None
    role_label: str
    start: float
    end: float
    start_label: str
    end_label: str
    text: str
    is_learner: bool
    issues: list[ReportIssueView] = field(default_factory=list)


@dataclass(frozen=True)
class ReportPatternView:
    category: str
    category_label: str
    label: str
    count: int
    examples: list[str]
    message: str


@dataclass(frozen=True)
class ReportView:
    """All data needed to render report.md / report.html."""

    source_audio: str
    audio_filename: str | None
    asr_model: str
    learner_speaker: str
    duration_seconds: float | None
    duration_label: str | None
    metrics: dict[str, float | int]
    long_pause_threshold_seconds: float
    report_turn_gap_seconds: float
    turns: list[ReportTurnView]
    has_feedback: bool
    summary: str | None
    recurring_patterns: list[ReportPatternView]
    issues: list[ReportIssueView]
    llm_provider: str | None
    llm_model: str | None


def _issue_view(issue: FeedbackIssue) -> ReportIssueView:
    return ReportIssueView(
        category=issue.category,
        category_label=category_label(issue.category),
        severity=issue.severity,
        severity_label=severity_label(issue.severity),
        scope=issue.scope,
        turn_index=issue.turn_index,
        start=issue.start,
        end=issue.end,
        start_label=format_timestamp(issue.start),
        end_label=format_timestamp(issue.end),
        excerpt=issue.excerpt,
        message=issue.message,
        suggestion=issue.suggestion,
    )


def _overlaps(issue: ReportIssueView, turn: DialogueTurn) -> bool:
    """True when the issue time range overlaps the display turn."""
    return issue.start < turn.end and issue.end > turn.start


def _turn_view(
    turn: DialogueTurn,
    *,
    learner_speaker: str,
    issue_views: list[ReportIssueView],
) -> ReportTurnView:
    return ReportTurnView(
        index=turn.index,
        speaker_id=turn.speaker_id,
        role=turn.role,
        role_label=role_label(turn.role),
        start=turn.start,
        end=turn.end,
        start_label=format_timestamp(turn.start),
        end_label=format_timestamp(turn.end),
        text=turn.text,
        is_learner=turn.speaker_id == learner_speaker,
        issues=[issue for issue in issue_views if _overlaps(issue, turn)],
    )


def build_report_view(
    transcript_document: TranscriptDocument,
    analysis_document: AnalysisDocument,
    feedback_document: FeedbackDocument | None = None,
    *,
    duration_seconds: float | None = None,
    audio_filename: str | None = None,
    turn_gap_seconds: float = DEFAULT_REPORT_TURN_GAP_SECONDS,
) -> ReportView:
    """Build a display-oriented report view from pipeline documents."""
    if turn_gap_seconds < 0:
        msg = "turn_gap_seconds must be >= 0"
        raise ValueError(msg)

    learner_speaker = analysis_document.learner_speaker
    turns = build_dialogue_turns(
        transcript_document.transcript,
        learner_speaker=learner_speaker,
        max_gap_seconds=turn_gap_seconds,
    )

    issue_views: list[ReportIssueView] = []
    patterns: list[ReportPatternView] = []
    summary: str | None = None
    llm_provider: str | None = None
    llm_model: str | None = None

    if feedback_document is not None:
        summary = feedback_document.summary
        llm_provider = feedback_document.llm_provider
        llm_model = feedback_document.llm_model
        for pattern in feedback_document.recurring_patterns:
            patterns.append(
                ReportPatternView(
                    category=pattern.category,
                    category_label=category_label(pattern.category),
                    label=pattern.label,
                    count=pattern.count,
                    examples=list(pattern.examples),
                    message=pattern.message,
                )
            )
        for issue in feedback_document.issues:
            issue_views.append(_issue_view(issue))

    turn_views = [
        _turn_view(
            turn,
            learner_speaker=learner_speaker,
            issue_views=issue_views,
        )
        for turn in turns
    ]

    metrics = analysis_document.metrics
    return ReportView(
        source_audio=transcript_document.source_audio,
        audio_filename=audio_filename or analysis_document.source_audio,
        asr_model=transcript_document.model,
        learner_speaker=learner_speaker,
        duration_seconds=duration_seconds,
        duration_label=(
            format_timestamp(duration_seconds) if duration_seconds is not None else None
        ),
        metrics={
            "speaking_time_seconds": metrics.speaking_time_seconds,
            "speaking_ratio": metrics.speaking_ratio,
            "wpm": metrics.wpm,
            "turn_count": metrics.turn_count,
            "average_turn_seconds": metrics.average_turn_seconds,
            "pause_count": metrics.pause_count,
            "long_pause_count": metrics.long_pause_count,
            "filler_count": metrics.filler_count,
        },
        long_pause_threshold_seconds=(analysis_document.long_pause_threshold_seconds),
        report_turn_gap_seconds=turn_gap_seconds,
        turns=turn_views,
        has_feedback=feedback_document is not None,
        summary=summary,
        recurring_patterns=patterns,
        issues=issue_views,
        llm_provider=llm_provider,
        llm_model=llm_model,
    )
