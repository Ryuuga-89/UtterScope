"""Homebrew-style step progress for the CLI."""

from __future__ import annotations

import threading

from utterscope.cli.console import (
    ANSI_CLEAR_LINE,
    ANSI_CYAN,
    ANSI_RESET,
    SPINNER_FRAMES,
    clear_current_line,
    console,
    spinner_supported,
    stream,
)


class CliProgress:
    """Show a spinner while a step runs, then replace it with ✔."""

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._step = ""

    def begin(self, step: str) -> None:
        self._stop_spinner()
        self._step = step
        if spinner_supported():
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._animate,
                name=f"utterscope-spinner-{step}",
                daemon=True,
            )
            self._thread.start()
            return
        console.print(f"  [cyan]⠋[/cyan] {step}...")

    def update(self, step: str) -> None:
        """Change the waiting label without finishing the current step."""
        self._step = step
        if not spinner_supported():
            console.print(f"  [cyan]⠋[/cyan] {step}...")

    def end(self, step: str, detail: str = "") -> None:
        self._stop_spinner()
        clear_current_line()
        suffix = f"  ({detail})" if detail else ""
        console.print(f"  [green]✔[/green] {step}{suffix}")

    def cancel(self) -> None:
        """Stop the spinner without a success mark (e.g. on error)."""
        self._stop_spinner()
        clear_current_line()

    def _stop_spinner(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._thread = None

    def _animate(self) -> None:
        out = stream()
        index = 0
        while not self._stop.is_set():
            frame = SPINNER_FRAMES[index % len(SPINNER_FRAMES)]
            out.write(
                f"\r{ANSI_CLEAR_LINE}  {ANSI_CYAN}{frame}{ANSI_RESET} {self._step}..."
            )
            out.flush()
            index += 1
            self._stop.wait(0.08)
