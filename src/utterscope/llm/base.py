"""LLM feedback backend protocol and errors."""

from __future__ import annotations

from typing import Protocol

from utterscope.llm.schemas import PassAResult, PassBResult
from utterscope.llm.turns import DialogueTurn


class LlmError(Exception):
    """Raised when LLM feedback cannot be produced."""


class FeedbackBackend(Protocol):
    """Two-pass language feedback backend."""

    def run_pass_a(
        self,
        turns: list[DialogueTurn],
        *,
        learner_speaker: str,
        model: str,
    ) -> PassAResult:
        """Whole-lesson summary and recurring patterns."""

    def run_pass_b(
        self,
        *,
        target: DialogueTurn,
        window: list[DialogueTurn],
        learner_speaker: str,
        model: str,
    ) -> PassBResult:
        """Per-turn issues for one learner turn."""
