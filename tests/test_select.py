"""Tests for interactive radio selection."""

from __future__ import annotations

from unittest.mock import patch

from rich.console import Console

from utterscope.cli.select import RadioOption, select_radio


def test_select_radio_single_option_returns_zero() -> None:
    console = Console(force_terminal=False)
    index = select_radio(
        title="Pick one",
        options=[RadioOption(label="only")],
        console=console,
    )
    assert index == 0


def test_select_radio_falls_back_to_numbered_prompt() -> None:
    console = Console(force_terminal=False)
    options = [
        RadioOption(label="A", detail="1s", samples=("hello",)),
        RadioOption(label="B", detail="2s", samples=("world",)),
    ]
    with (
        patch("sys.stdin.isatty", return_value=False),
        patch("typer.prompt", return_value=2),
    ):
        index = select_radio(
            title="Which?",
            options=options,
            console=console,
        )
    assert index == 1


def test_select_radio_keyboard_navigation() -> None:
    console = Console(force_terminal=True)
    options = [
        RadioOption(label="A"),
        RadioOption(label="B"),
        RadioOption(label="C"),
    ]
    keys = iter(["\x1b[B", "\x1bOB", "\r"])
    with (
        patch("sys.stdin.isatty", return_value=True),
        patch("utterscope.cli.select._read_key", side_effect=lambda: next(keys)),
    ):
        index = select_radio(
            title="Which?",
            options=options,
            console=console,
        )
    assert index == 2
