"""``utterscope history`` — list indexed analyze runs."""

from __future__ import annotations

from rich.table import Table

from utterscope.cli.console import console
from utterscope.config import resolve_db_path
from utterscope.storage import list_runs


def run_history(*, limit: int = 20) -> None:
    """Print recent lesson history as a Rich table."""
    if limit < 0:
        msg = "limit must be >= 0"
        raise ValueError(msg)

    db_path = resolve_db_path()
    runs = list_runs(limit=limit, db_path=db_path)
    if not runs:
        console.print("[dim]履歴はありません[/dim]")
        console.print(f"[dim]DB → {db_path}[/dim]")
        return

    table = Table(
        title="Lesson history",
        show_header=True,
        header_style="bold",
    )
    table.add_column("日時", style="cyan", no_wrap=True)
    table.add_column("Run")
    table.add_column("WPM", justify="right")
    table.add_column("発話比", justify="right")
    table.add_column("フィラー", justify="right")
    table.add_column("指摘", justify="right")
    table.add_column("ディレクトリ", overflow="fold")

    for run in runs:
        wpm = f"{run.wpm:.0f}" if run.wpm is not None else "—"
        ratio = f"{run.speaking_ratio:.0%}" if run.speaking_ratio is not None else "—"
        fillers = str(run.filler_count) if run.filler_count is not None else "—"
        issues = str(run.issue_count) if run.llm_enabled else "—"
        table.add_row(
            run.created_at,
            run.run_name,
            wpm,
            ratio,
            fillers,
            issues,
            run.run_dir,
        )

    console.print(table)
    console.print(f"[dim]DB → {db_path}[/dim]")
