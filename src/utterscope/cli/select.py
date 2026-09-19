"""Interactive radio-style selection for the CLI."""

from __future__ import annotations

import os
import select
import sys
import termios
import tty
from dataclasses import dataclass
from typing import TextIO, cast

import typer
from rich.console import Console

from utterscope.cli.console import (
    ANSI_BOLD,
    ANSI_CLEAR_LINE,
    ANSI_CYAN,
    ANSI_DIM,
    ANSI_HIDE_CURSOR,
    ANSI_RESET,
    ANSI_SHOW_CURSOR,
)
from utterscope.cli.console import (
    console as default_console,
)

_KEY_UP = frozenset({"\x1b[A", "\x1bOA", "k", "K"})
_KEY_DOWN = frozenset({"\x1b[B", "\x1bOB", "j", "J"})
_KEY_ENTER = frozenset({"\r", "\n"})
_KEY_CANCEL = frozenset({"q", "Q", "\x1b"})


@dataclass(frozen=True)
class RadioOption:
    """One selectable row in a radio menu."""

    label: str
    detail: str = ""
    samples: tuple[str, ...] = ()


def select_radio(
    *,
    title: str,
    options: list[RadioOption],
    console: Console | None = None,
) -> int:
    """Return the selected option index.

    On an interactive TTY, navigate with ↑/↓ (or j/k) and confirm with Enter.
    Otherwise fall back to a numbered prompt.
    """
    ui = console or default_console
    if not options:
        msg = "no options to select"
        raise ValueError(msg)
    if len(options) == 1:
        return 0
    if sys.stdin.isatty() and ui.is_terminal:
        return _select_with_keys(title=title, options=options, console=ui)
    return _select_with_numbers(title=title, options=options, console=ui)


def _select_with_keys(
    *,
    title: str,
    options: list[RadioOption],
    console: Console,
) -> int:
    out = cast(TextIO, console.file)
    width = max(40, (console.width or 80) - 1)
    index = 0
    height = 0

    out.write(ANSI_HIDE_CURSOR)
    out.flush()
    try:
        height = _draw_menu(
            out,
            title=title,
            options=options,
            index=index,
            width=width,
            previous_height=0,
        )
        while True:
            key = _read_key()
            if key in _KEY_ENTER:
                break
            if key == "\x03":
                raise KeyboardInterrupt
            if key in _KEY_CANCEL:
                raise typer.Abort()

            previous = index
            if key in _KEY_UP:
                index = (index - 1) % len(options)
            elif key in _KEY_DOWN:
                index = (index + 1) % len(options)
            elif key.isdigit():
                choice = int(key)
                if 1 <= choice <= len(options):
                    index = choice - 1
                    height = _draw_menu(
                        out,
                        title=title,
                        options=options,
                        index=index,
                        width=width,
                        previous_height=height,
                    )
                    break
            if index == previous:
                continue
            height = _draw_menu(
                out,
                title=title,
                options=options,
                index=index,
                width=width,
                previous_height=height,
            )
    finally:
        out.write(ANSI_SHOW_CURSOR)
        out.flush()

    return index


def _select_with_numbers(
    *,
    title: str,
    options: list[RadioOption],
    console: Console,
) -> int:
    console.print()
    console.print(f"[bold]{title}[/bold]")
    for i, option in enumerate(options, start=1):
        detail = f"  ({option.detail})" if option.detail else ""
        console.print(f"\n  [{i}] {option.label}{detail}")
        for sample in option.samples:
            console.print(f"      • {sample}")
    while True:
        choice = typer.prompt("Enter number", type=int)
        if 1 <= choice <= len(options):
            return choice - 1
        console.print("[red]Invalid choice. Try again.[/red]")


def _draw_menu(
    out: TextIO,
    *,
    title: str,
    options: list[RadioOption],
    index: int,
    width: int,
    previous_height: int,
) -> int:
    lines = _menu_lines(title, options, index, width)
    if previous_height > 0:
        out.write(f"\033[{previous_height}A")

    for line in lines:
        out.write(f"\r{ANSI_CLEAR_LINE}{line}\n")

    for _ in range(len(lines), previous_height):
        out.write(f"\r{ANSI_CLEAR_LINE}\n")
    if previous_height > len(lines):
        out.write(f"\033[{previous_height - len(lines)}A")

    out.flush()
    return len(lines)


def _menu_lines(
    title: str,
    options: list[RadioOption],
    index: int,
    width: int,
) -> list[str]:
    lines = [f"{ANSI_BOLD}{title}{ANSI_RESET}", ""]
    for i, option in enumerate(options):
        selected = i == index
        mark = "●" if selected else "○"
        style = f"{ANSI_BOLD}{ANSI_CYAN}" if selected else ANSI_DIM
        head = f"  {mark} {option.label}"
        if option.detail:
            head += f"  ({option.detail})"
        lines.append(f"{style}{_truncate(head, width)}{ANSI_RESET}")
        for sample in option.samples:
            lines.append(f"{style}{_truncate(f'      {sample}', width)}{ANSI_RESET}")
        lines.append("")
    lines.append(f"{ANSI_DIM}  ↑/↓ 移動  ·  Enter 決定{ANSI_RESET}")
    return lines


def _truncate(text: str, width: int) -> str:
    if width <= 1:
        return "…"
    if len(text) <= width:
        return text
    return text[: width - 1] + "…"


def _read_key() -> str:
    """Read one keypress, including CSI/SS3 arrow sequences."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        first = _read_byte(fd)
        if first is None:
            return "\x03"
        if first != "\x1b":
            return first

        second = _read_byte(fd, timeout=0.1)
        if second is None:
            return first
        if second in {"[", "O"}:
            third = _read_byte(fd, timeout=0.1)
            if third is None:
                return first + second
            return first + second + third
        return first + second
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _read_byte(fd: int, timeout: float | None = None) -> str | None:
    if timeout is not None:
        ready, _, _ = select.select([fd], [], [], timeout)
        if not ready:
            return None
    data = os.read(fd, 1)
    if not data:
        return None
    return data.decode("latin-1")
