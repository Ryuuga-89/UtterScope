"""CLI application entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel

from utterscope import __version__
from utterscope.audio import AudioPreparationError
from utterscope.models import AnalyzeRequest, AnalyzeResult
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


def format_duration(seconds: float) -> str:
    total = int(round(seconds))
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {secs:02d}s"
    return f"{minutes}m {secs:02d}s"


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

    header = f"[bold]{audio.name}[/bold]"
    console.print(Panel.fit(header, title="UtterScope", border_style="cyan"))

    if request.llm:
        console.print(
            "[dim]Note: --llm is accepted but ignored in v0.1.[/dim]"
        )

    try:
        result = run_pipeline(request)
    except AudioPreparationError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(EXIT_RUNTIME_ERROR) from exc
    except OSError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(EXIT_USER_ERROR) from exc
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(EXIT_RUNTIME_ERROR) from exc

    _print_summary(result)


def _print_summary(result: AnalyzeResult) -> None:
    if result.duration_seconds is not None:
        console.print(
            f"  Duration: {format_duration(result.duration_seconds)}"
        )
        console.print()

    console.print("  ✓ prepare audio")
    console.print("  · detect speech          [dim]n/a (v0.2)[/dim]")
    console.print("  · transcribe             [yellow]pending[/yellow]")
    console.print("  · identify speakers      [dim]n/a (v0.2)[/dim]")
    console.print("  · analyze learner speech [dim]n/a (v0.2)[/dim]")
    console.print()
    console.print(
        "[yellow]Pipeline stub:[/yellow] transcription is not implemented yet."
    )
    console.print(f"Intended transcript → {result.transcript_path}")
