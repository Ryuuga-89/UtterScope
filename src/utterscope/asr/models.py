"""Map CLI model names to mlx-whisper Hugging Face repos."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

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

# Shown in interactive ASR selection (skip alias duplicates like turbo).
ASR_MODEL_CHOICES: tuple[str, ...] = (
    "tiny",
    "tiny.en",
    "base",
    "small",
    "medium",
    "large",
    "large-v2",
    "large-v3",
    "large-v3-turbo",
)

DEFAULT_ASR_MODEL = "tiny.en"


@dataclass(frozen=True)
class AsrModelInfo:
    """ASR model row for interactive selection."""

    name: str
    repo: str
    downloaded: bool

    @property
    def status_label(self) -> str:
        return "downloaded" if self.downloaded else "not downloaded"


def resolve_model_path(model: str) -> str:
    """Resolve a short model name or pass through a full repo/path."""
    if "/" in model or model.startswith("."):
        return model
    return _MODEL_ALIASES.get(model, model)


def huggingface_hub_cache_dir() -> Path:
    """Return the Hugging Face Hub cache directory used for model blobs."""
    for key in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        raw = os.environ.get(key)
        if raw:
            return Path(raw).expanduser()
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        return Path(HF_HUB_CACHE)
    except ImportError:
        hf_home = os.environ.get("HF_HOME")
        if hf_home:
            return Path(hf_home).expanduser() / "hub"
        return Path.home() / ".cache" / "huggingface" / "hub"


def is_model_downloaded(model: str, *, cache_dir: Path | None = None) -> bool:
    """Return True when ``model`` appears present in the local HF hub cache."""
    resolved = resolve_model_path(model)
    local = Path(resolved).expanduser()
    if local.is_dir():
        return True
    if "/" not in resolved:
        return False

    root = cache_dir or huggingface_hub_cache_dir()
    dirname = "models--" + resolved.replace("/", "--")
    snapshots = root / dirname / "snapshots"
    if not snapshots.is_dir():
        return False
    try:
        return any(snapshots.iterdir())
    except OSError:
        return False


def list_asr_models(*, cache_dir: Path | None = None) -> list[AsrModelInfo]:
    """Return selectable ASR models with local download status."""
    return [
        AsrModelInfo(
            name=name,
            repo=resolve_model_path(name),
            downloaded=is_model_downloaded(name, cache_dir=cache_dir),
        )
        for name in ASR_MODEL_CHOICES
    ]
