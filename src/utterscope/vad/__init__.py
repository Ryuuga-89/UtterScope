"""Voice activity detection."""

from utterscope.vad.base import VadBackend
from utterscope.vad.silero import SileroVadBackend, VadError
from utterscope.vad.types import SpeechInterval, VadResult

__all__ = [
    "SileroVadBackend",
    "SpeechInterval",
    "VadBackend",
    "VadError",
    "VadResult",
]
