"""Environment and local configuration helpers."""

from __future__ import annotations

import os
import re
from pathlib import Path

from dotenv import load_dotenv

from utterscope.models import DEFAULT_LONG_PAUSE_THRESHOLD_SECONDS

_ENV_LINE_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")

HF_TOKEN_KEY = "HF_TOKEN"
LONG_PAUSE_THRESHOLD_KEY = "UTTERSCOPE_LONG_PAUSE_THRESHOLD"


def project_env_path(project_dir: Path | None = None) -> Path:
    """Return the path to the project ``.env`` file."""
    directory = project_dir or Path.cwd()
    return directory / ".env"


def load_project_env(project_dir: Path | None = None) -> Path | None:
    """Load ``.env`` from the project directory if present.

    Existing process environment variables are not overridden, so an explicit
    ``export HF_TOKEN=...`` still wins over ``.env``.
    """
    env_path = project_env_path(project_dir)
    if not env_path.is_file():
        return None
    load_dotenv(env_path, override=False)
    return env_path


def read_env_value(key: str, project_dir: Path | None = None) -> str | None:
    """Read ``key`` from process env first, then from ``.env`` if present."""
    existing = os.environ.get(key)
    if existing is not None and existing != "":
        return existing

    env_path = project_env_path(project_dir)
    if not env_path.is_file():
        return None
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = _ENV_LINE_RE.match(stripped)
        if match and match.group(1) == key:
            value = match.group(2).strip().strip("'").strip('"')
            return value or None
    return None


def upsert_env_value(
    key: str,
    value: str,
    project_dir: Path | None = None,
) -> Path:
    """Create or update ``key=value`` in the project ``.env`` file.

    Other lines (including comments) are preserved. The process environment is
    updated for the current run as well.
    """
    if not key.isidentifier():
        msg = f"invalid environment key: {key!r}"
        raise ValueError(msg)

    env_path = project_env_path(project_dir)
    lines: list[str] = []
    if env_path.is_file():
        lines = env_path.read_text(encoding="utf-8").splitlines()

    assignment = f"{key}={value}"
    replaced = False
    updated: list[str] = []
    for line in lines:
        match = _ENV_LINE_RE.match(line.strip()) if line.strip() else None
        if match and match.group(1) == key:
            updated.append(assignment)
            replaced = True
        else:
            updated.append(line)

    if not replaced:
        if updated and updated[-1] != "":
            updated.append("")
        updated.append(assignment)

    text = "\n".join(updated).rstrip() + "\n"
    env_path.write_text(text, encoding="utf-8")
    os.environ[key] = value
    return env_path


def mask_secret(value: str, *, visible: int = 4) -> str:
    """Return a masked preview that never echoes the full secret."""
    if not value:
        return "(empty)"
    if len(value) <= visible:
        return "*" * len(value)
    return f"{'*' * max(8, len(value) - visible)}{value[-visible:]}"


def resolve_long_pause_threshold(
    cli_value: float | None = None,
    *,
    project_dir: Path | None = None,
) -> float:
    """Resolve long-pause threshold: CLI > env/.env > default."""
    if cli_value is not None:
        if cli_value <= 0:
            msg = "long pause threshold must be greater than 0"
            raise ValueError(msg)
        return cli_value

    raw = read_env_value(LONG_PAUSE_THRESHOLD_KEY, project_dir)
    if raw is None:
        return DEFAULT_LONG_PAUSE_THRESHOLD_SECONDS
    try:
        value = float(raw)
    except ValueError as exc:
        msg = (
            f"invalid {LONG_PAUSE_THRESHOLD_KEY}={raw!r}; "
            "expected a positive number"
        )
        raise ValueError(msg) from exc
    if value <= 0:
        msg = f"{LONG_PAUSE_THRESHOLD_KEY} must be greater than 0"
        raise ValueError(msg)
    return value
