"""Tests for per-run output directories."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from utterscope.config import (
    OUTPUT_ASK_KEY,
    OUTPUT_ROOT_KEY,
    resolve_output_root,
)
from utterscope.output import (
    create_run_layout,
    next_run_serial,
    sanitize_audio_stem,
)


def test_sanitize_audio_stem_strips_unsafe_chars() -> None:
    assert sanitize_audio_stem("my lesson (1)") == "my_lesson_1"
    assert sanitize_audio_stem("...") == "audio"


def test_allocate_run_dirs_use_date_serial_and_stem(tmp_path: Path) -> None:
    audio = tmp_path / "test0.mp3"
    audio.write_bytes(b"x")
    root = tmp_path / "results"
    now = datetime(2025, 9, 19, 12, 0, tzinfo=ZoneInfo("Asia/Tokyo"))

    first = create_run_layout(root, audio, now=now)
    second = create_run_layout(root, audio, now=now)

    assert first.run_dir.name == "250919-1_test0"
    assert second.run_dir.name == "250919-2_test0"
    assert first.source_audio.name == "test0.mp3"
    assert first.source_audio.is_file()
    assert first.source_audio.read_bytes() == b"x"


def test_next_run_serial_ignores_other_stems(tmp_path: Path) -> None:
    (tmp_path / "250919-3_other").mkdir()
    (tmp_path / "250919-1_test0").mkdir()
    assert next_run_serial(tmp_path, date_prefix="250919", audio_stem="test0") == 2


def test_resolve_output_root_defaults_to_cwd_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(OUTPUT_ROOT_KEY, raising=False)
    monkeypatch.delenv(OUTPUT_ASK_KEY, raising=False)
    root = resolve_output_root(None, interactive=False, project_dir=tmp_path)
    assert root == (tmp_path / "results").resolve()


def test_resolve_output_root_requires_absolute_configured_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(OUTPUT_ROOT_KEY, "relative/results")
    monkeypatch.delenv(OUTPUT_ASK_KEY, raising=False)
    with pytest.raises(ValueError, match="absolute"):
        resolve_output_root(None, interactive=False, project_dir=tmp_path)


def test_resolve_output_root_ask_errors_in_non_interactive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(OUTPUT_ASK_KEY, "1")
    with pytest.raises(ValueError, match="OUTPUT_ASK"):
        resolve_output_root(None, interactive=False, project_dir=tmp_path)


def test_resolve_output_root_ask_prompts_when_interactive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(OUTPUT_ASK_KEY, "1")
    chosen = tmp_path / "custom"
    root = resolve_output_root(
        None,
        interactive=True,
        project_dir=tmp_path,
        prompt_absolute_path=lambda: str(chosen),
    )
    assert root == chosen.resolve()
