"""Tests for FFmpeg audio preparation."""

from __future__ import annotations

from pathlib import Path

import pytest

from utterscope.audio import (
    PREPARED_FILENAME,
    TARGET_CHANNELS,
    TARGET_SAMPLE_RATE,
    AudioPreparationError,
    prepare_audio,
)


def test_prepare_audio_converts_to_16k_mono(sample_audio: Path, tmp_path: Path) -> None:
    work_dir = tmp_path / "work"
    prepared = prepare_audio(sample_audio, work_dir)

    assert prepared.path == work_dir / PREPARED_FILENAME
    assert prepared.path.is_file()
    assert prepared.sample_rate == TARGET_SAMPLE_RATE
    assert prepared.channels == TARGET_CHANNELS
    assert prepared.duration_seconds == pytest.approx(1.0, abs=0.1)


def test_prepare_audio_rejects_invalid_file(tmp_path: Path) -> None:
    source = tmp_path / "broken.mp3"
    source.write_bytes(b"not-audio")

    with pytest.raises(AudioPreparationError):
        prepare_audio(source, tmp_path / "work")
