"""ASR backend protocol."""

from __future__ import annotations

from typing import Protocol

from utterscope.audio import PreparedAudio
from utterscope.models import Transcript


class AsrBackend(Protocol):
    """Speech recognition backend."""

    def transcribe(self, audio: PreparedAudio, *, model: str) -> Transcript:
        """Transcribe prepared audio into a timestamped transcript."""
