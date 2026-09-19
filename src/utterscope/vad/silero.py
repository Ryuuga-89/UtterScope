"""Silero VAD backend."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from utterscope.audio import PreparedAudio
from utterscope.vad.types import SpeechInterval, VadResult


class VadError(Exception):
    """Raised when voice activity detection fails."""


class SileroVadBackend:
    """VAD backend backed by ``silero-vad``."""

    def detect(self, audio: PreparedAudio) -> VadResult:
        try:
            from silero_vad import get_speech_timestamps, load_silero_vad
        except ImportError as exc:
            msg = (
                "silero-vad is not installed; "
                "run `uv sync` to enable speech detection"
            )
            raise VadError(msg) from exc

        try:
            model = load_silero_vad()
            waveform = _read_prepared_waveform(audio.path)
            timestamps = get_speech_timestamps(
                waveform,
                model,
                sampling_rate=audio.sample_rate,
                return_seconds=True,
            )
        except VadError:
            raise
        except Exception as exc:
            msg = f"silero-vad detection failed: {exc}"
            raise VadError(msg) from exc

        intervals = [
            SpeechInterval(start=float(item["start"]), end=float(item["end"]))
            for item in timestamps
        ]
        return VadResult(intervals=intervals)


def _read_prepared_waveform(path: Path):
    """Load a mono PCM WAV into a 1-D float32 torch tensor."""
    import torch

    with wave.open(str(path), "rb") as handle:
        channel_count = handle.getnchannels()
        sample_width = handle.getsampwidth()
        frame_count = handle.getnframes()
        frames = handle.readframes(frame_count)

    if channel_count != 1:
        msg = f"expected mono audio, got {channel_count} channels"
        raise VadError(msg)
    if sample_width != 2:
        msg = f"expected 16-bit PCM audio, got sample width {sample_width}"
        raise VadError(msg)

    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    return torch.from_numpy(samples)
