"""CLI application entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from utterscope import __version__

app = typer.Typer(
    name="utterscope",
    help="Local-first speaking analysis for language learners.",
    no_args_is_help=True,
)
console = Console()


def version_callback(value: bool) -> None:
    if value:
        console.print(f"UtterScope {__version__}")
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
        typer.Argument(help="Path to the audio file to analyze.", exists=True),
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
    output_dir = output or Path.cwd()
    console.print(f"[bold]UtterScope[/bold] {__version__}")
    console.print(f"Audio:  {audio}")
    console.print(f"Model:  {model}")
    console.print(f"LLM:    {'on' if llm else 'off'}")
    console.print(f"Output: {output_dir}")
    console.print(
        "\n[yellow]Pipeline is not implemented yet "
        "(project skeleton only).[/yellow]"
    )
