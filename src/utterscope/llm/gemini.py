"""Gemini API backend for language feedback."""

from __future__ import annotations

import os
from typing import TypeVar

from pydantic import BaseModel

from utterscope.config import GEMINI_API_KEY
from utterscope.llm.base import LlmError
from utterscope.llm.prompts import build_pass_a_prompt, build_pass_b_prompt
from utterscope.llm.schemas import PassAResult, PassBResult
from utterscope.llm.turns import DialogueTurn

T = TypeVar("T", bound=BaseModel)


class GeminiFeedbackBackend:
    """Feedback backend backed by the Google Gen AI SDK (Gemini)."""

    def __init__(self, *, api_key: str | None = None) -> None:
        self._api_key = api_key or os.environ.get(GEMINI_API_KEY)
        if not self._api_key:
            msg = (
                f"{GEMINI_API_KEY} is not set. "
                "Run `utterscope setup` or export the key, "
                "or pass --no-llm to skip feedback."
            )
            raise LlmError(msg)

    def run_pass_a(
        self,
        turns: list[DialogueTurn],
        *,
        learner_speaker: str,
        model: str,
    ) -> PassAResult:
        prompt = build_pass_a_prompt(turns, learner_speaker=learner_speaker)
        return self._generate(model=model, prompt=prompt, schema=PassAResult)

    def run_pass_b(
        self,
        *,
        target: DialogueTurn,
        window: list[DialogueTurn],
        learner_speaker: str,
        model: str,
    ) -> PassBResult:
        prompt = build_pass_b_prompt(
            target=target,
            window=window,
            learner_speaker=learner_speaker,
        )
        return self._generate(model=model, prompt=prompt, schema=PassBResult)

    def _generate(
        self,
        *,
        model: str,
        prompt: str,
        schema: type[T],
    ) -> T:
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            msg = "google-genai is not installed; run `uv sync` to enable LLM feedback"
            raise LlmError(msg) from exc

        client = genai.Client(api_key=self._api_key)
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema,
                ),
            )
        except Exception as exc:
            msg = f"Gemini request failed: {exc}"
            raise LlmError(msg) from exc

        parsed = getattr(response, "parsed", None)
        if parsed is not None:
            if isinstance(parsed, schema):
                return parsed
            return schema.model_validate(parsed)

        text = getattr(response, "text", None)
        if not text:
            msg = "Gemini returned an empty response"
            raise LlmError(msg)
        try:
            return schema.model_validate_json(text)
        except Exception as exc:
            msg = f"Gemini returned invalid JSON for {schema.__name__}: {exc}"
            raise LlmError(msg) from exc
