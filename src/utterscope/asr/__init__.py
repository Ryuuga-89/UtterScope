"""Automatic speech recognition backends."""

from utterscope.asr.base import AsrBackend
from utterscope.asr.convert import transcript_from_whisper_result
from utterscope.asr.mlx_backend import AsrError, MlxWhisperBackend
from utterscope.asr.models import resolve_model_path

__all__ = [
    "AsrBackend",
    "AsrError",
    "MlxWhisperBackend",
    "resolve_model_path",
    "transcript_from_whisper_result",
]
