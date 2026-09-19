"""Interactive radio-style selection for the CLI."""

from __future__ import annotations

import os
import re
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
_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


@dataclass(frozen=True)
class RadioOption:
    """One selectable row in a radio menu."""

    label: str
    detail: str = ""
    samples: tuple[str, ...] = ()
    # Optional truecolor/ANSI prefixes applied to label / detail text.
    label_ansi: str = ""
    detail_ansi: str = ""


def select_radio(
    *,
    title: str,
    options: list[RadioOption],
    default_index: int = 0,
    subtitle: str = "",
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
    if not 0 <= default_index < len(options):
        msg = f"default_index out of range: {default_index}"
        raise ValueError(msg)
    if sys.stdin.isatty() and ui.is_terminal:
        return _select_with_keys(
            title=title,
            options=options,
            default_index=default_index,
            subtitle=subtitle,
            console=ui,
        )
    return _select_with_numbers(
        title=title,
        options=options,
        default_index=default_index,
        subtitle=subtitle,
        console=ui,
    )


def _select_with_keys(
    *,
    title: str,
    options: list[RadioOption],
    default_index: int,
    subtitle: str,
    console: Console,
) -> int:
    out = cast(TextIO, console.file)
    width = max(40, (console.width or 80) - 1)
    index = default_index
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
            subtitle=subtitle,
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
                        subtitle=subtitle,
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
                subtitle=subtitle,
            )
    finally:
        out.write(ANSI_SHOW_CURSOR)
        out.flush()

    return index


def _select_with_numbers(
    *,
    title: str,
    options: list[RadioOption],
    default_index: int,
    subtitle: str,
    console: Console,
) -> int:
    console.print()
    console.print(f"[bold]{title}[/bold]")
    if subtitle:
        console.print(f"[dim]{subtitle}[/dim]")
    for i, option in enumerate(options, start=1):
        detail = f"  {option.detail}" if option.detail else ""
        console.print(f"\n  [{i}] {option.label}{detail}")
        for sample in option.samples:
            console.print(f"      • {sample}")
    default_choice = default_index + 1
    while True:
        choice = typer.prompt(
            "Enter number",
            type=int,
            default=default_choice,
        )
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
    subtitle: str = "",
) -> int:
    lines = _menu_lines(title, options, index, width, subtitle=subtitle)
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
    *,
    subtitle: str = "",
) -> list[str]:
    lines = [f"{ANSI_BOLD}{title}{ANSI_RESET}"]
    if subtitle:
        lines.append(f"{ANSI_DIM}{subtitle}{ANSI_RESET}")
    lines.append("")
    for i, option in enumerate(options):
        selected = i == index
        mark = "●" if selected else "○"
        mark_style = f"{ANSI_BOLD}{ANSI_CYAN}" if selected else ANSI_DIM
        if selected and option.label_ansi:
            label_style = f"{ANSI_BOLD}{option.label_ansi}"
        elif option.label_ansi:
            label_style = option.label_ansi
        elif selected:
            label_style = f"{ANSI_BOLD}{ANSI_CYAN}"
        else:
            label_style = ANSI_DIM

        head = (
            f"{mark_style}  {mark}{ANSI_RESET} {label_style}{option.label}{ANSI_RESET}"
        )
        if option.detail:
            detail_style = option.detail_ansi or (ANSI_DIM if not selected else "")
            if detail_style:
                head += f"  {detail_style}{option.detail}{ANSI_RESET}"
            else:
                head += f"  {option.detail}"
        lines.append(_truncate_ansi(head, width))
        for sample in option.samples:
            sample_style = f"{ANSI_BOLD}{ANSI_CYAN}" if selected else ANSI_DIM
            lines.append(
                _truncate_ansi(
                    f"{sample_style}      {sample}{ANSI_RESET}",
                    width,
                )
            )
        lines.append("")
    lines.append(f"{ANSI_DIM}  ↑/↓ 移動  ·  Enter 決定{ANSI_RESET}")
    return lines


def _truncate(text: str, width: int) -> str:
    if width <= 1:
        return "…"
    if len(text) <= width:
        return text
    return text[: width - 1] + "…"


def _truncate_ansi(text: str, width: int) -> str:
    """Truncate ``text`` by visible characters, preserving ANSI codes."""
    if width <= 1:
        return "…"
    if _visible_len(text) <= width:
        return text

    result: list[str] = []
    visible = 0
    i = 0
    while i < len(text):
        if text.startswith("\033[", i):
            end = text.find("m", i)
            if end == -1:
                result.append(text[i:])
                break
            result.append(text[i : end + 1])
            i = end + 1
            continue
        if visible >= width - 1:
            result.append("…")
            break
        result.append(text[i])
        visible += 1
        i += 1
    result.append(ANSI_RESET)
    return "".join(result)


def _visible_len(text: str) -> int:
    return len(_ANSI_RE.sub("", text))


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
