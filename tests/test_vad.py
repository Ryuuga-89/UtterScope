"""Tests for voice activity detection."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from utterscope.audio import PreparedAudio, prepare_audio
from utterscope.vad import SileroVadBackend, SpeechInterval, VadResult


def test_speech_interval_rejects_inverted_range() -> None:
    with pytest.raises(ValueError):
        SpeechInterval(start=2.0, end=1.0)


def test_silero_vad_maps_timestamps(
    sample_audio: Path, tmp_path: Path
) -> None:
    prepared = prepare_audio(sample_audio, tmp_path / "work")
    fake_timestamps = [{"start": 0.1, "end": 0.8}]

    with (
        patch("silero_vad.load_silero_vad", return_value=MagicMock()),
        patch(
            "silero_vad.get_speech_timestamps",
            return_value=fake_timestamps,
        ),
    ):
        result = SileroVadBackend().detect(prepared)

    assert result == VadResult(
        intervals=[SpeechInterval(start=0.1, end=0.8)],
    )


def test_silero_vad_runs_on_prepared_audio(
    sample_audio: Path, tmp_path: Path
) -> None:
    prepared = prepare_audio(sample_audio, tmp_path / "work")
    result = SileroVadBackend().detect(prepared)

    assert isinstance(result, VadResult)
    assert isinstance(prepared, PreparedAudio)
