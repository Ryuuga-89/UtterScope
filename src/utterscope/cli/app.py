"""CLI application entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel

from utterscope import __version__
from utterscope.models import AnalyzeRequest
from utterscope.pipeline import run as run_pipeline

app = typer.Typer(
    name="utterscope",
    help="Local-first speaking analysis for language learners.",
    no_args_is_help=True,
)
console = Console(stderr=True)

EXIT_USER_ERROR = 1
EXIT_RUNTIME_ERROR = 2


def version_callback(value: bool) -> None:
    if value:
        # Version is useful on stdout for scripting.
        typer.echo(f"UtterScope {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            "-V",
            help="Show version and exit.",
            callback=version_callback,
            is_eager=True,
        ),
    ] = False,
) -> None:
    """UtterScope CLI."""


@app.command()
def analyze(
    audio: Annotated[
        Path,
        typer.Argument(
            help="Path to the audio file to analyze.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ],
    model: Annotated[
        str,
        typer.Option("--model", help="ASR model to use."),
    ] = "large-v3-turbo",
    llm: Annotated[
        bool,
        typer.Option("--llm/--no-llm", help="Enable optional LLM analysis."),
    ] = True,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Directory for analysis outputs."),
    ] = None,
) -> None:
    """Analyze a recorded lesson or conversation."""
    output_dir = (output or Path.cwd()).resolve()
    request = AnalyzeRequest(
        audio_path=audio,
        model=model,
        output_dir=output_dir,
        llm=llm,
    )

    console.print(
        Panel.fit(
            f"[bold]{audio.name}[/bold]",
            title="UtterScope",
            border_style="cyan",
        )
    )

    if request.llm:
        console.print(
            "[dim]Note: --llm is accepted but ignored in v0.1.[/dim]"
        )

    try:
        result = run_pipeline(request)
    except OSError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(EXIT_USER_ERROR) from exc
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(EXIT_RUNTIME_ERROR) from exc

    console.print("  · prepare audio          [yellow]pending[/yellow]")
    console.print("  · detect speech          [dim]n/a (v0.2)[/dim]")
    console.print("  · transcribe             [yellow]pending[/yellow]")
    console.print("  · identify speakers      [dim]n/a (v0.2)[/dim]")
    console.print("  · analyze learner speech [dim]n/a (v0.2)[/dim]")
    console.print()
    console.print(
        "[yellow]Pipeline stub:[/yellow] transcription is not implemented yet."
    )
    console.print(f"Intended transcript → {result.transcript_path}")
