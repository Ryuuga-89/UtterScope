"""Map CLI model names to mlx-whisper Hugging Face repos."""

from __future__ import annotations

_MODEL_ALIASES: dict[str, str] = {
    "tiny": "mlx-community/whisper-tiny",
    "tiny.en": "mlx-community/whisper-tiny.en-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large": "mlx-community/whisper-large-mlx",
    "large-v2": "mlx-community/whisper-large-v2-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    "turbo": "mlx-community/whisper-large-v3-turbo",
}


def resolve_model_path(model: str) -> str:
    """Resolve a short model name or pass through a full repo/path."""
    if "/" in model or model.startswith("."):
        return model
    return _MODEL_ALIASES.get(model, model)
