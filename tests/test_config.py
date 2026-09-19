"""Tests for project .env loading and updates."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from utterscope.config import (
    LONG_PAUSE_THRESHOLD_KEY,
    load_project_env,
    mask_secret,
    read_env_value,
    resolve_long_pause_threshold,
    upsert_env_value,
)
from utterscope.models import DEFAULT_LONG_PAUSE_THRESHOLD_SECONDS


def test_load_project_env_sets_missing_values(tmp_path: Path, monkeypatch) -> None:
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


def test_upsert_env_value_creates_and_updates(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    path = upsert_env_value("HF_TOKEN", "token-one", tmp_path)
    assert path.read_text(encoding="utf-8") == "HF_TOKEN=token-one\n"
    assert os.environ["HF_TOKEN"] == "token-one"

    (tmp_path / ".env").write_text(
        "# keep me\nHF_TOKEN=token-one\nOTHER=1\n",
        encoding="utf-8",
    )
    upsert_env_value("HF_TOKEN", "token-two", tmp_path)
    text = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "# keep me" in text
    assert "HF_TOKEN=token-two" in text
    assert "OTHER=1" in text
    assert os.environ["HF_TOKEN"] == "token-two"
    monkeypatch.delenv("HF_TOKEN", raising=False)


def test_read_env_value_and_mask(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    (tmp_path / ".env").write_text("HF_TOKEN=abcdefghij\n", encoding="utf-8")
    assert read_env_value("HF_TOKEN", tmp_path) == "abcdefghij"
    assert mask_secret("abcdefghij").endswith("ghij")
    assert "*" in mask_secret("abcdefghij")


def test_resolve_long_pause_threshold_priority(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv(LONG_PAUSE_THRESHOLD_KEY, raising=False)
    assert (
        resolve_long_pause_threshold(project_dir=tmp_path)
        == DEFAULT_LONG_PAUSE_THRESHOLD_SECONDS
    )

    upsert_env_value(LONG_PAUSE_THRESHOLD_KEY, "1.5", tmp_path)
    assert resolve_long_pause_threshold(project_dir=tmp_path) == 1.5
    assert resolve_long_pause_threshold(2.0, project_dir=tmp_path) == 2.0

    with pytest.raises(ValueError):
        resolve_long_pause_threshold(0, project_dir=tmp_path)
    monkeypatch.delenv(LONG_PAUSE_THRESHOLD_KEY, raising=False)
