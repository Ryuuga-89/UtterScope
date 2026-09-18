"""MLX Whisper ASR backend."""

from __future__ import annotations

from utterscope.asr.convert import transcript_from_whisper_result
from utterscope.asr.models import resolve_model_path
from utterscope.audio import PreparedAudio
from utterscope.models import Transcript


class AsrError(Exception):
    """Raised when speech recognition fails."""


class MlxWhisperBackend:
    """ASR backend backed by ``mlx-whisper``."""

    def transcribe(self, audio: PreparedAudio, *, model: str) -> Transcript:
        try:
            import mlx_whisper
        except ImportError as exc:
            msg = (
                "mlx-whisper is not installed; "
                "run `uv sync` on Apple Silicon to enable transcription"
            )
            raise AsrError(msg) from exc

        model_path = resolve_model_path(model)
        try:
            result = mlx_whisper.transcribe(
                str(audio.path),
                path_or_hf_repo=model_path,
                verbose=False,
            )
        except Exception as exc:
            msg = f"mlx-whisper transcription failed: {exc}"
            raise AsrError(msg) from exc

        if not isinstance(result, dict):
            msg = "mlx-whisper returned an unexpected result type"
            raise AsrError(msg)

        return transcript_from_whisper_result(result)
