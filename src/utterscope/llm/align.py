"""Attach Pass B drafts to dialogue turns and drop invalid excerpts."""

from __future__ import annotations

from utterscope.llm.schemas import PassBIssue
from utterscope.llm.turns import DialogueTurn
from utterscope.models import FeedbackIssue


def issues_for_turn(
    drafts: list[PassBIssue],
    *,
    turn: DialogueTurn,
) -> list[FeedbackIssue]:
    """Convert Pass B drafts into timed FeedbackIssue rows for ``turn``."""
    issues: list[FeedbackIssue] = []
    normalized_turn = _normalize(turn.text)
    for draft in drafts:
        if not _excerpt_fits(draft.excerpt, normalized_turn):
            continue
        issues.append(
            FeedbackIssue(
                category=draft.category,
                severity=draft.severity,
                scope=draft.scope,
                turn_index=turn.index,
                start=turn.start,
                end=turn.end,
                excerpt=draft.excerpt.strip(),
                message=draft.message.strip(),
                suggestion=(draft.suggestion.strip() if draft.suggestion else None),
            )
        )
    return issues


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _excerpt_fits(excerpt: str, normalized_turn: str) -> bool:
    needle = _normalize(excerpt)
    if not needle:
        return False
    return needle in normalized_turn
