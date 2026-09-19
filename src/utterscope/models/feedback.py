"""LLM feedback document models written to feedback.json."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

FEEDBACK_SCHEMA_VERSION = 1
DEFAULT_LLM_PROVIDER = "gemini"
DEFAULT_LLM_MODEL = "gemini-3.8-flash"
DEFAULT_FEEDBACK_CONTEXT_TURNS = 1

LlmProviderName = Literal["gemini"]
FeedbackCategoryName = Literal["grammar", "naturalness", "vocabulary"]
FeedbackSeverityName = Literal["low", "medium", "high"]
FeedbackScopeName = Literal["turn", "phrase"]


class FeedbackIssue(BaseModel):
    """One actionable finding anchored to a learner turn.

    Pass B produces these. ``scope="turn"`` is whole-turn feedback (level B);
    ``scope="phrase"`` targets an excerpt within the turn (level C).
    Word-level (level D) is out of scope for v0.3.
    """

    category: FeedbackCategoryName
    severity: FeedbackSeverityName = "medium"
    scope: FeedbackScopeName = "turn"
    turn_index: int = Field(
        ge=0,
        description="Index into the chronological dialogue turn list.",
    )
    start: float = Field(ge=0, description="Playback start time in seconds.")
    end: float = Field(ge=0, description="Playback end time in seconds.")
    excerpt: str = Field(
        min_length=1,
        description="Quote from the learner turn (prefer verbatim ASR text).",
    )
    message: str = Field(min_length=1, description="What is wrong or weak.")
    suggestion: str | None = Field(
        default=None,
        description="Clearer or more natural alternative, when applicable.",
    )

    @model_validator(mode="after")
    def validate_time_range(self) -> FeedbackIssue:
        if self.end < self.start:
            msg = "end must be greater than or equal to start"
            raise ValueError(msg)
        return self


class RecurringPattern(BaseModel):
    """A repeated tendency observed across the lesson (Pass A)."""

    category: FeedbackCategoryName
    label: str = Field(min_length=1, description="Short pattern name.")
    count: int = Field(ge=1, description="Estimated occurrences.")
    examples: list[str] = Field(default_factory=list)
    message: str = Field(min_length=1)


class FeedbackDocument(BaseModel):
    """Serializable LLM feedback artifact written to feedback.json."""

    schema_version: int = FEEDBACK_SCHEMA_VERSION
    source_audio: str
    asr_model: str = Field(description="Whisper (or other ASR) model id used.")
    llm_provider: LlmProviderName = DEFAULT_LLM_PROVIDER
    llm_model: str = Field(
        default=DEFAULT_LLM_MODEL,
        description="Gemini model id used for feedback.",
    )
    learner_speaker: str
    summary: str = Field(
        min_length=1,
        description="Required whole-lesson overview from Pass A.",
    )
    recurring_patterns: list[RecurringPattern] = Field(default_factory=list)
    issues: list[FeedbackIssue] = Field(default_factory=list)
    context_turns: int = Field(
        default=DEFAULT_FEEDBACK_CONTEXT_TURNS,
        ge=0,
        description=(
            "Number of neighboring turns included as context in Pass B "
            "(initial default: 1)."
        ),
    )
