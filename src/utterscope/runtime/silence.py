"""Helpers to keep third-party library output out of the CLI."""

from __future__ import annotations

import os
import warnings
from collections.abc import Iterator
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path

_QUIET_ENV = {
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "TQDM_DISABLE": "1",
    "TRANSFORMERS_VERBOSITY": "error",
    "TOKENIZERS_PARALLELISM": "false",
}


def enable_quiet_mode() -> None:
    """Prefer quiet defaults for Hugging Face / tqdm / warnings."""
    for key, value in _QUIET_ENV.items():
        os.environ.setdefault(key, value)
    warnings.filterwarnings("ignore")


@contextmanager
def silence_third_party(*, enabled: bool = True) -> Iterator[None]:
    """Suppress stdout/stderr noise from ML backends while ``enabled``."""
    if not enabled:
        yield
        return

    enable_quiet_mode()
    with Path(os.devnull).open("w", encoding="utf-8") as devnull:
        with (
            warnings.catch_warnings(),
            redirect_stdout(devnull),
            redirect_stderr(devnull),
        ):
            warnings.simplefilter("ignore")
            yield
