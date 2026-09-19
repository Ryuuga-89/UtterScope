"""Diarization backend protocol."""

from __future__ import annotations

from typing import Protocol

from utterscope.audio import PreparedAudio
from utterscope.diarization.types import DiarizationResult


class DiarizationBackend(Protocol):
    """Speaker diarization backend."""

    def diarize(
        self,
        audio: PreparedAudio,
        *,
        num_speakers: int | None = 2,
    ) -> DiarizationResult:
        """Return speaker turns for prepared audio."""
