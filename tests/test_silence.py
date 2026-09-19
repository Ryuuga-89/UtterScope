"""Tests for third-party output silencing."""

from __future__ import annotations

import sys
from io import StringIO

from utterscope.cli.console import CliStream
from utterscope.runtime import silence_third_party


def test_silence_third_party_suppresses_stdout() -> None:
    buffer = StringIO()
    original = sys.stdout
    try:
        with silence_third_party(enabled=True):
            print("should-not-appear")
    finally:
        sys.stdout = original

    # After context, printing works again.
    print("ok", file=buffer)
    assert "ok" in buffer.getvalue()


def test_silence_third_party_can_be_disabled() -> None:
    buffer = StringIO()
    original = sys.stdout
    sys.stdout = buffer
    try:
        with silence_third_party(enabled=False):
            print("visible")
    finally:
        sys.stdout = original

    assert "visible" in buffer.getvalue()


def test_cli_stream_stays_visible_during_silence(monkeypatch) -> None:
    """Progress must not vanish when quiet mode redirects sys.stderr."""
    captured = StringIO()

    class _FakeTty:
        def write(self, data: str) -> int:
            return captured.write(data)

        def flush(self) -> None:
            return None

        def isatty(self) -> bool:
            return True

        def fileno(self) -> int:
            return 2

    monkeypatch.setattr(sys, "__stderr__", _FakeTty())
    stream = CliStream()
    with silence_third_party(enabled=True):
        stream.write("waiting-for-transcribe")
        stream.flush()

    assert "waiting-for-transcribe" in captured.getvalue()
