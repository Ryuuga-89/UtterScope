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
    label: str = Field(min_length=1)
    count: int = Field(ge=1)
    examples: list[str] = Field(default_factory=list)
    message: str = Field(min_length=1)


class PassAResult(BaseModel):
    """Whole-lesson overview produced by Pass A."""

    summary: str = Field(min_length=1)
    recurring_patterns: list[PassAPattern] = Field(default_factory=list)


class PassBIssue(BaseModel):
    """Issue draft from Pass B; turn timing is attached by the runner."""

    category: FeedbackCategoryName
    severity: FeedbackSeverityName = "medium"
    scope: FeedbackScopeName = "turn"
    excerpt: str = Field(min_length=1)
    message: str = Field(min_length=1)
    suggestion: str | None = None


class PassBResult(BaseModel):
    """Per-turn findings produced by Pass B."""

    issues: list[PassBIssue] = Field(default_factory=list)
