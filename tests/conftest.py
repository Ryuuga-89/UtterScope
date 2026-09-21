"""Shared pytest fixtures."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from utterscope.config import DB_PATH_KEY


@pytest.fixture(autouse=True)
def isolate_history_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep lesson-history writes inside the test temp directory."""
    monkeypatch.setenv(DB_PATH_KEY, str(tmp_path / "history.sqlite"))


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
