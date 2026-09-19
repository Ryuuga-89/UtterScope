"""Per-run output directory allocation under the configured root."""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

_UNSAFE_RE = re.compile(r"[^\w\-]+", re.UNICODE)
_RUN_DIR_RE = re.compile(r"^(\d{6})-(\d+)_(.+)$")


@dataclass(frozen=True)
class RunLayout:
    """Paths for one analyze run directory."""

    root: Path
    run_dir: Path
    source_audio: Path


def sanitize_audio_stem(stem: str) -> str:
    """Make an audio stem safe for use in a directory name (no extension)."""
    cleaned = _UNSAFE_RE.sub("_", stem).strip("._")
    return cleaned or "audio"


def next_run_serial(
    root: Path,
    *,
    date_prefix: str,
    audio_stem: str,
) -> int:
    """Return the next serial ``n`` for ``YYMMDD-n_stem`` under ``root``."""
    if not root.is_dir():
        return 1
    highest = 0
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        match = _RUN_DIR_RE.match(entry.name)
        if match is None:
            continue
        if match.group(1) != date_prefix:
            continue
        if match.group(3) != audio_stem:
            continue
        highest = max(highest, int(match.group(2)))
    return highest + 1


def allocate_run_dir(
    root: Path,
    audio_path: Path,
    *,
    now: datetime | None = None,
) -> Path:
    """Create ``YYMMDD-n_<stem>`` under ``root`` and return it."""
    moment = now or datetime.now().astimezone()
    date_prefix = moment.strftime("%y%m%d")
    stem = sanitize_audio_stem(audio_path.stem)
    root.mkdir(parents=True, exist_ok=True)
    serial = next_run_serial(root, date_prefix=date_prefix, audio_stem=stem)
    run_dir = root / f"{date_prefix}-{serial}_{stem}"
    run_dir.mkdir(parents=False, exist_ok=False)
    return run_dir


def create_run_layout(
    root: Path,
    audio_path: Path,
    *,
    now: datetime | None = None,
) -> RunLayout:
    """Allocate a run directory and copy the source audio into it."""
    audio = audio_path.expanduser().resolve()
    if not audio.is_file():
        msg = f"audio file not found: {audio}"
        raise FileNotFoundError(msg)

    run_dir = allocate_run_dir(root, audio, now=now)
    destination = run_dir / audio.name
    shutil.copy2(audio, destination)
    return RunLayout(root=root.resolve(), run_dir=run_dir, source_audio=destination)
