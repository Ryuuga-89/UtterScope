"""Automatic speech recognition backends."""

from utterscope.asr.base import AsrBackend
from utterscope.asr.convert import transcript_from_whisper_result
from utterscope.asr.mlx_backend import AsrError, MlxWhisperBackend
from utterscope.asr.models import (
    ASR_MODEL_CHOICES,
    DEFAULT_ASR_MODEL,
    is_model_downloaded,
    list_asr_models,
    resolve_model_path,
)

__all__ = [
    "ASR_MODEL_CHOICES",
    "AsrBackend",
    "AsrError",
    "DEFAULT_ASR_MODEL",
    "MlxWhisperBackend",
    "is_model_downloaded",
    "list_asr_models",
    "resolve_model_path",
    "transcript_from_whisper_result",
]
