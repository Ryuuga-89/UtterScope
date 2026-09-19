"""Catalog of LLM providers and selectable models."""

from __future__ import annotations

from dataclasses import dataclass

from utterscope.models import DEFAULT_LLM_MODEL, DEFAULT_LLM_PROVIDER


@dataclass(frozen=True)
class LlmProviderInfo:
    """One LLM provider offered in interactive mode."""

    id: str
    label: str
    detail: str = ""


@dataclass(frozen=True)
class LlmModelInfo:
    """One provider-specific model offered in interactive mode."""

    id: str
    family: str
    detail: str = ""

    @property
    def label(self) -> str:
        return self.id


LLM_PROVIDERS: tuple[LlmProviderInfo, ...] = (
    LlmProviderInfo(
        id=DEFAULT_LLM_PROVIDER,
        label="Gemini",
        detail="Google AI Studio",
    ),
)

# A few Flash-Lite / Flash / Pro options for interactive selection.
GEMINI_MODELS: tuple[LlmModelInfo, ...] = (
    LlmModelInfo("gemini-2.5-flash-lite", "Flash Lite", "fast · low cost"),
    LlmModelInfo("gemini-3.1-flash-lite", "Flash Lite", "3.1"),
    LlmModelInfo("gemini-3.5-flash-lite", "Flash Lite", "3.5 · latest lite"),
    LlmModelInfo("gemini-2.5-flash", "Flash", "2.5"),
    LlmModelInfo("gemini-3.5-flash", "Flash", "3.5"),
    LlmModelInfo("gemini-3.8-flash", "Flash", "3.8 · default"),
    LlmModelInfo("gemini-2.5-pro", "Pro", "2.5"),
    LlmModelInfo("gemini-3.1-pro-preview", "Pro", "3.1 · preview"),
)


def default_gemini_model_index(
    *,
    model_id: str = DEFAULT_LLM_MODEL,
) -> int:
    """Return catalog index for ``model_id``, or 0 when unknown."""
    for index, model in enumerate(GEMINI_MODELS):
        if model.id == model_id:
            return index
    return 0


def models_for_provider(provider_id: str) -> tuple[LlmModelInfo, ...]:
    """Return selectable models for ``provider_id``."""
    if provider_id == DEFAULT_LLM_PROVIDER:
        return GEMINI_MODELS
    msg = f"unsupported LLM provider: {provider_id}"
    raise ValueError(msg)
