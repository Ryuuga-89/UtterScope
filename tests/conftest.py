"""Shared pytest fixtures."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def sample_audio(tmp_path: Path) -> Path:
    """Create a short stereo WAV for preprocessing tests."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is required for audio fixtures")

    path = tmp_path / "lesson.wav"
    command = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:duration=1",
        "-ar",
        "44100",
        "-ac",
        "2",
        str(path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        pytest.fail(f"failed to create sample audio: {completed.stderr}")
    return path
