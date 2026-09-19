"""Tests for project .env loading."""

from __future__ import annotations

import os
from pathlib import Path

from utterscope.config import load_project_env


def test_load_project_env_sets_missing_values(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("UTTERSCOPE_TEST_TOKEN", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("UTTERSCOPE_TEST_TOKEN=from-dotenv\n", encoding="utf-8")

    loaded = load_project_env(tmp_path)

    assert loaded == env_file
    assert os.environ["UTTERSCOPE_TEST_TOKEN"] == "from-dotenv"
    monkeypatch.delenv("UTTERSCOPE_TEST_TOKEN", raising=False)


def test_load_project_env_does_not_override_existing(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("UTTERSCOPE_TEST_TOKEN", "from-shell")
    env_file = tmp_path / ".env"
    env_file.write_text("UTTERSCOPE_TEST_TOKEN=from-dotenv\n", encoding="utf-8")

    loaded = load_project_env(tmp_path)

    assert loaded == env_file
    assert os.environ["UTTERSCOPE_TEST_TOKEN"] == "from-shell"


def test_load_project_env_missing_file(tmp_path: Path) -> None:
    assert load_project_env(tmp_path) is None
