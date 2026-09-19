"""Orchestrate Pass A / Pass B language feedback."""

from __future__ import annotations

from collections.abc import Callable

from utterscope.llm.align import issues_for_turn
from utterscope.llm.base import FeedbackBackend, LlmError
from utterscope.llm.gemini import GeminiFeedbackBackend
from utterscope.llm.turns import (
    build_dialogue_turns,
    context_window,
    learner_turn_indices,
)
from utterscope.models import (
    DEFAULT_FEEDBACK_CONTEXT_TURNS,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_PROVIDER,
    FeedbackDocument,
    FeedbackIssue,
    RecurringPattern,
    Transcript,
)

ProgressCallback = Callable[[str], None]

__all__ = [
    "FeedbackBackend",
    "GeminiFeedbackBackend",
    "LlmError",
    "generate_feedback_document",
]


def generate_feedback_document(
    transcript: Transcript,
    *,
    learner_speaker: str,
    source_audio: str,
    asr_model: str,
    llm_model: str = DEFAULT_LLM_MODEL,
    context_turns: int = DEFAULT_FEEDBACK_CONTEXT_TURNS,
    backend: FeedbackBackend | None = None,
    on_progress: ProgressCallback | None = None,
) -> FeedbackDocument:
    """Run two-pass feedback and return a FeedbackDocument."""
    turns = build_dialogue_turns(
        transcript,
        learner_speaker=learner_speaker,
    )
    if not turns:
        msg = "no speaker-labeled turns available for LLM feedback"
        raise LlmError(msg)

    client = backend or GeminiFeedbackBackend()
    if on_progress is not None:
        on_progress("pass A: lesson overview")
    pass_a = client.run_pass_a(
        turns,
        learner_speaker=learner_speaker,
        model=llm_model,
    )
    if not pass_a.summary.strip():
        msg = "Gemini Pass A returned an empty summary"
        raise LlmError(msg)

    learner_indices = learner_turn_indices(
        turns,
        learner_speaker=learner_speaker,
    )
    issues: list[FeedbackIssue] = []
    total = len(learner_indices)
    for position, turn_index in enumerate(learner_indices, start=1):
        target = turns[turn_index]
        if on_progress is not None:
            on_progress(f"pass B: learner turn {position}/{total}")
        window = context_window(
            turns,
            center_index=turn_index,
            radius=context_turns,
        )
        pass_b = client.run_pass_b(
            target=target,
            window=window,
            learner_speaker=learner_speaker,
            model=llm_model,
        )
        issues.extend(issues_for_turn(pass_b.issues, turn=target))

    return FeedbackDocument(
        source_audio=source_audio,
        asr_model=asr_model,
        llm_provider=DEFAULT_LLM_PROVIDER,
        llm_model=llm_model,
        learner_speaker=learner_speaker,
        summary=pass_a.summary.strip(),
        recurring_patterns=[
            RecurringPattern(
                category=pattern.category,
                label=pattern.label,
                count=pattern.count,
                examples=list(pattern.examples),
                message=pattern.message,
            )
            for pattern in pass_a.recurring_patterns
        ],
        issues=issues,
        context_turns=context_turns,
    )
