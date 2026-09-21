"""Persistent storage (SQLite lesson history)."""

from utterscope.storage.db import HistoryRun, connect, list_runs, record_run

__all__ = [
    "HistoryRun",
    "connect",
    "list_runs",
    "record_run",
]
