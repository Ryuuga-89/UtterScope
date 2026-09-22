"""Tests for interactive setup."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from utterscope.cli.app import app
from utterscope.cli.setup_cmd import run_setup
from utterscope.config import (
    GEMINI_API_KEY,
    LONG_PAUSE_THRESHOLD_KEY,
    REPORT_TURN_GAP_KEY,
)

runner = CliRunner()


def test_setup_saves_token_and_threshold(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv(GEMINI_API_KEY, raising=False)
    monkeypatch.delenv(LONG_PAUSE_THRESHOLD_KEY, raising=False)
    # Menu: token → gemini → threshold → done (Done is index 5)
    choices = iter([0, 1, 2, 5])
    prompts = iter(["hf_test_token", "gemini_test_key", "1.5"])

    with (
        patch(
            "utterscope.cli.setup_cmd.select_radio",
            side_effect=lambda **_: next(choices),
        ),
        patch("typer.prompt", side_effect=lambda *a, **k: next(prompts)),
    ):
        written = run_setup(project_dir=tmp_path)

    text = written.read_text(encoding="utf-8")
    assert "HF_TOKEN=hf_test_token" in text
    assert f"{GEMINI_API_KEY}=gemini_test_key" in text
    assert f"{LONG_PAUSE_THRESHOLD_KEY}=1.5" in text


def test_setup_saves_report_turn_gap(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv(REPORT_TURN_GAP_KEY, raising=False)
    choices = iter([3, 5])
    prompts = iter(["8"])

    with (
        patch(
            "utterscope.cli.setup_cmd.select_radio",
            side_effect=lambda **_: next(choices),
        ),
        patch("typer.prompt", side_effect=lambda *a, **k: next(prompts)),
    ):
        written = run_setup(project_dir=tmp_path)

    assert f"{REPORT_TURN_GAP_KEY}=8" in written.read_text(encoding="utf-8")


def test_setup_can_finish_without_changes(tmp_path: Path) -> None:
    with patch("utterscope.cli.setup_cmd.select_radio", return_value=5):
        written = run_setup(project_dir=tmp_path)
    assert written == tmp_path / ".env"


def test_setup_command_is_registered() -> None:
    result = runner.invoke(app, ["setup", "--help"])
    assert result.exit_code == 0
    assert ".env" in result.output
