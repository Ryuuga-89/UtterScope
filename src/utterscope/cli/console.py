"""Shared CLI console and terminal helpers."""

from __future__ import annotations

import os
import sys
from typing import TextIO, cast

from rich.console import Console

# Raw ANSI for spinner / in-place redraw (Rich markup strips bare escapes).
ANSI_HIDE_CURSOR = "\033[?25l"
ANSI_SHOW_CURSOR = "\033[?25h"
ANSI_CLEAR_LINE = "\033[2K"
ANSI_BOLD = "\033[1m"
ANSI_DIM = "\033[2m"
ANSI_CYAN = "\033[36m"
ANSI_YELLOW = "\033[33m"
ANSI_RESET = "\033[0m"

SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def ansi_rgb(r: int, g: int, b: int) -> str:
    """Truecolor foreground escape for ``(r, g, b)``."""
    return f"\033[38;2;{r};{g};{b}m"


class CliStream:
    """Prefer the real terminal stderr so quiet-mode redirects stay invisible.

    When stderr is not a TTY (e.g. CliRunner), follow ``sys.stderr`` so test
    harnesses can still capture CLI output.
    """

    def write(self, data: str) -> int:
        return self._target().write(data)

    def flush(self) -> None:
        self._target().flush()

    def isatty(self) -> bool:
        return self._target().isatty()

    def fileno(self) -> int:
        return self._target().fileno()

    @staticmethod
    def _target() -> TextIO:
        real = sys.__stderr__
        if real is not None and real.isatty():
            return real
        return sys.stderr


console = Console(file=cast(TextIO, CliStream()))


def stream() -> TextIO:
    """Return the console output stream (quiet-mode safe)."""
    return cast(TextIO, console.file)


def spinner_supported() -> bool:
    """True when an animated spinner can be shown."""
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return False
    return stream().isatty()


def clear_current_line() -> None:
    if not spinner_supported():
        return
    out = stream()
    out.write(f"\r{ANSI_CLEAR_LINE}")
    out.flush()
