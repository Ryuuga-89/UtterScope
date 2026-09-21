"""Record a completed analyze run into the lesson-history index."""

from __future__ import annotations

from utterscope.cli.console import console
from utterscope.models import AnalyzeResult
from utterscope.pipeline.analyze import PipelineProgress
from utterscope.storage import record_run


def record_history(
    result: AnalyzeResult,
    *,
    progress: PipelineProgress | None = None,
) -> None:
    """Best-effort history write; never fails the analyze run."""
    if progress is not None:
        progress.begin("record history")
    try:
        db_path = record_run(result)
    except Exception as exc:
        if progress is not None:
            progress.cancel()
        console.print(f"[yellow]Warning:[/yellow] could not record history: {exc}")
        return
    if progress is not None:
        progress.end("record history", db_path.name)
