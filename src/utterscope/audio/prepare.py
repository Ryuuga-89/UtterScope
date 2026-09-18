"""FFmpeg-based audio preprocessing."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from utterscope.audio.types import PreparedAudio

TARGET_SAMPLE_RATE = 16_000
TARGET_CHANNELS = 1
PREPARED_FILENAME = "prepared.wav"


class AudioPreparationError(Exception):
    """Raised when audio cannot be prepared for analysis."""


def prepare_audio(source: Path, work_dir: Path) -> PreparedAudio:
    """Convert ``source`` into 16 kHz mono PCM WAV under ``work_dir``."""
    ffmpeg = _require_executable("ffmpeg")
    ffprobe = _require_executable("ffprobe")

    work_dir.mkdir(parents=True, exist_ok=True)
    output_path = work_dir / PREPARED_FILENAME

    command = [
        ffmpeg,
        "-y",
        "-i",
        str(source),
        "-ac",
        str(TARGET_CHANNELS),
        "-ar",
        str(TARGET_SAMPLE_RATE),
        "-c:a",
        "pcm_s16le",
        "-vn",
        str(output_path),
    ]
    _run(command, action="convert audio")

    duration_seconds = _probe_duration(ffprobe, output_path)
    return PreparedAudio(
        path=output_path,
        sample_rate=TARGET_SAMPLE_RATE,
        channels=TARGET_CHANNELS,
        duration_seconds=duration_seconds,
    )


def _require_executable(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        msg = f"{name} was not found on PATH; install FFmpeg to continue"
        raise AudioPreparationError(msg)
    return path


def _run(command: list[str], *, action: str) -> None:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        msg = f"failed to {action}: {exc}"
        raise AudioPreparationError(msg) from exc

    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        msg = f"failed to {action}"
        if detail:
            msg = f"{msg}: {detail}"
        raise AudioPreparationError(msg)


def _probe_duration(ffprobe: str, path: Path) -> float:
    command = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        msg = f"failed to probe duration: {exc}"
        raise AudioPreparationError(msg) from exc

    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        msg = "failed to probe duration"
        if detail:
            msg = f"{msg}: {detail}"
        raise AudioPreparationError(msg)

    try:
        payload = json.loads(completed.stdout)
        duration = float(payload["format"]["duration"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        msg = "ffprobe returned an unexpected duration payload"
        raise AudioPreparationError(msg) from exc

    if duration < 0:
        msg = "ffprobe returned a negative duration"
        raise AudioPreparationError(msg)
    return duration
