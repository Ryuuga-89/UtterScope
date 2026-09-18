"""Audio preprocessing (FFmpeg)."""

from utterscope.audio.prepare import (
    PREPARED_FILENAME,
    TARGET_CHANNELS,
    TARGET_SAMPLE_RATE,
    AudioPreparationError,
    prepare_audio,
)
from utterscope.audio.types import PreparedAudio

__all__ = [
    "PREPARED_FILENAME",
    "TARGET_CHANNELS",
    "TARGET_SAMPLE_RATE",
    "AudioPreparationError",
    "PreparedAudio",
    "prepare_audio",
]
