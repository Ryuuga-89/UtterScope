"""LLM response schemas for Pass A / Pass B (not the on-disk document)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from utterscope.models.feedback import (
    FeedbackCategoryName,
    FeedbackScopeName,
    FeedbackSeverityName,
)


class PassAPattern(BaseModel):
    category: FeedbackCategoryName
    label: str = Field(
        min_length=1,
        description="Short pattern name in Japanese.",
    )
    count: int = Field(ge=1)
    examples: list[str] = Field(
        default_factory=list,
        description="Verbatim English quotes from the learner.",
    )
    message: str = Field(
        min_length=1,
        description="Explanation of the pattern in Japanese.",
    )


class PassAResult(BaseModel):
    """Whole-lesson overview produced by Pass A."""

    summary: str = Field(
        min_length=1,
        description="Whole-lesson overview written in Japanese.",
    )
    recurring_patterns: list[PassAPattern] = Field(default_factory=list)


class PassBIssue(BaseModel):
    """Issue draft from Pass B; turn timing is attached by the runner."""

    category: FeedbackCategoryName
    severity: FeedbackSeverityName = "medium"
    scope: FeedbackScopeName = "turn"
    excerpt: str = Field(
        min_length=1,
        description="Verbatim English quote from the target turn.",
    )
    message: str = Field(
        min_length=1,
        description="What is wrong or weak, written in Japanese.",
    )
    suggestion: str | None = Field(
        default=None,
        description=(
            "Improved English alternative when helpful (may keep English wording)."
        ),
    )


class PassBResult(BaseModel):
    """Per-turn findings produced by Pass B."""

    issues: list[PassBIssue] = Field(default_factory=list)
