"""VAD backend protocol."""

from __future__ import annotations

from typing import Protocol

from utterscope.audio import PreparedAudio
from utterscope.vad.types import VadResult


class VadBackend(Protocol):
    """Voice activity detection backend."""

    def detect(self, audio: PreparedAudio) -> VadResult:
        """Return speech intervals for prepared audio."""
