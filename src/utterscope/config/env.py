"""Environment and local configuration helpers."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv


def load_project_env(project_dir: Path | None = None) -> Path | None:
    """Load ``.env`` from the project directory if present.

    Existing process environment variables are not overridden, so an explicit
    ``export HF_TOKEN=...`` still wins over ``.env``.
    """
    directory = project_dir or Path.cwd()
    env_path = directory / ".env"
    if not env_path.is_file():
        return None
    load_dotenv(env_path, override=False)
    return env_path
