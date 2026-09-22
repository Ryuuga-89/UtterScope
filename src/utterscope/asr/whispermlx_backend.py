"""whispermlx ASR + alignment + speaker diarization backend."""

from __future__ import annotations

import logging
import os
from typing import Any

from utterscope.asr.convert import transcript_from_whisper_result
from utterscope.audio import PreparedAudio
from utterscope.models import Transcript

logger = logging.getLogger(__name__)

DEFAULT_LANGUAGE = "en"
DEFAULT_TORCH_DEVICE = "mps"
DEFAULT_BATCH_SIZE = 16
DEFAULT_NUM_SPEAKERS = 2


class AsrError(Exception):
    """Raised when speech recognition or diarization fails."""


class WhisperMlxBackend:
    """Combined transcription + diarization via ``whispermlx``.

    ASR runs on Apple Silicon through mlx-whisper (Metal). Alignment and
    pyannote diarization use the torch device (default ``mps``).
    """

    def __init__(
        self,
        *,
        language: str = DEFAULT_LANGUAGE,
        torch_device: str = DEFAULT_TORCH_DEVICE,
        batch_size: int = DEFAULT_BATCH_SIZE,
        num_speakers: int = DEFAULT_NUM_SPEAKERS,
    ) -> None:
        self._language = language
        self._torch_device = torch_device
        self._batch_size = batch_size
        self._num_speakers = num_speakers

    def transcribe(self, audio: PreparedAudio, *, model: str) -> Transcript:
        try:
            import whispermlx
        except ImportError as exc:
            msg = (
                "whispermlx is not installed; "
                "run `uv sync` on Apple Silicon to enable transcription"
            )
            raise AsrError(msg) from exc

        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        if not token:
            msg = (
                "Hugging Face token is required for speaker diarization. "
                "Set HF_TOKEN in .env after accepting model terms for "
                "pyannote/speaker-diarization-community-1"
            )
            raise AsrError(msg)

        audio_path = str(audio.path.resolve())
        device = self._resolve_torch_device()

        try:
            waveform = whispermlx.load_audio(audio_path)
            asr_model = whispermlx.load_model(
                model,
                device,
                language=self._language,
            )
            result = asr_model.transcribe(waveform, batch_size=self._batch_size)
            language = result.get("language") or self._language

            align_model, metadata = whispermlx.load_align_model(
                language_code=language,
                device=device,
            )
            result = whispermlx.align(
                result["segments"],
                align_model,
                metadata,
                waveform,
                device,
                return_char_alignments=False,
            )

            diarize_model = self._load_diarization_pipeline(token, device)
            diarize_segments = diarize_model(
                waveform,
                min_speakers=self._num_speakers,
                max_speakers=self._num_speakers,
            )
            result = whispermlx.assign_word_speakers(diarize_segments, result)
        except AsrError:
            raise
        except Exception as exc:
            msg = f"whispermlx transcription failed: {exc}"
            raise AsrError(msg) from exc

        if not isinstance(result, dict):
            msg = "whispermlx returned an unexpected result type"
            raise AsrError(msg)

        payload: dict[str, Any] = dict(result)
        if payload.get("language") is None:
            payload["language"] = self._language

        return transcript_from_whisper_result(payload)

    def _resolve_torch_device(self) -> str:
        preferred = self._torch_device
        if preferred != "mps":
            return preferred
        try:
            import torch

            if torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        logger.warning("MPS unavailable; falling back to cpu for torch stages")
        return "cpu"

    def _load_diarization_pipeline(self, token: str, device: str) -> Any:
        from whispermlx.diarize import DiarizationPipeline

        last_error: Exception | None = None
        for kwargs in (
            {"token": token, "device": device},
            {"use_auth_token": token, "device": device},
            {"token": token},
        ):
            try:
                return DiarizationPipeline(**kwargs)
            except TypeError as exc:
                last_error = exc
                continue
        msg = f"Could not construct DiarizationPipeline: {last_error}"
        raise AsrError(msg)
