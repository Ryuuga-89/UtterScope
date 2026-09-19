"""CLI application entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.panel import Panel

from utterscope import __version__
from utterscope.asr import AsrError
from utterscope.audio import AudioPreparationError
from utterscope.cli.console import console
from utterscope.cli.learner import InteractiveLearnerSelector
from utterscope.cli.progress import CliProgress
from utterscope.config import load_project_env
from utterscope.diarization import DiarizationError, FixedLearnerSelector
from utterscope.models import AnalyzeRequest
from utterscope.pipeline import run as run_pipeline
from utterscope.runtime import enable_quiet_mode
from utterscope.vad import VadError

# Load project .env early so backends can read HF_TOKEN and similar secrets.
load_project_env()

app = typer.Typer(
    name="utterscope",
    help="Local-first speaking analysis for language learners.",
    no_args_is_help=True,
)

EXIT_USER_ERROR = 1
EXIT_RUNTIME_ERROR = 2

_PIPELINE_ERRORS = (
    AudioPreparationError,
    VadError,
    DiarizationError,
    AsrError,
)


def version_callback(value: bool) -> None:
    if value:
        typer.echo(f"UtterScope {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    ctx: typer.Context,
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
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Show third-party library logs and warnings.",
        ),
    ] = False,
) -> None:
    """UtterScope CLI."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    if not verbose:
        enable_quiet_mode()


@app.command()
def analyze(
    ctx: typer.Context,
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
    learner: Annotated[
        str | None,
        typer.Option(
            "--learner",
            help="Learner speaker id (skips interactive selection).",
        ),
    ] = None,
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
        learner_speaker=learner,
    )
    verbose = bool(ctx.obj.get("verbose", False))

    console.print(
        Panel.fit(
            f"[bold]{audio.name}[/bold]",
            title="UtterScope",
            border_style="cyan",
        )
    )
    if request.llm:
        console.print(
            "[dim]Note: --llm is accepted but ignored until LLM analysis lands.[/dim]"
        )

    selector = (
        FixedLearnerSelector(learner)
        if learner is not None
        else InteractiveLearnerSelector()
    )
    progress = CliProgress()
    try:
        result = run_pipeline(
            request,
            progress=progress,
            learner_selector=selector,
            quiet=not verbose,
        )
    except (typer.Abort, KeyboardInterrupt) as exc:
        progress.cancel()
        console.print("\n[dim]Cancelled.[/dim]")
        raise typer.Exit(EXIT_USER_ERROR) from exc
    except _PIPELINE_ERRORS as exc:
        progress.cancel()
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(EXIT_RUNTIME_ERROR) from exc
    except OSError as exc:
        progress.cancel()
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(EXIT_USER_ERROR) from exc
    except Exception as exc:
        progress.cancel()
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(EXIT_RUNTIME_ERROR) from exc

    console.print("  [dim]○[/dim] analyze learner speech  [dim]pending[/dim]")
    console.print()
    console.print(f"Transcript → {result.transcript_path}")
    if result.learner_speaker is not None:
        console.print(f"Learner → {result.learner_speaker}")
