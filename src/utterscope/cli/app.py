"""CLI application entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.panel import Panel

from utterscope import __version__
from utterscope.asr import AsrError
from utterscope.asr.models import DEFAULT_ASR_MODEL
from utterscope.audio import AudioPreparationError
from utterscope.cli.console import console
from utterscope.cli.interactive import run_interactive
from utterscope.cli.learner import InteractiveLearnerSelector
from utterscope.cli.progress import CliProgress
from utterscope.cli.setup_cmd import run_setup
from utterscope.config import (
    GEMINI_API_KEY,
    load_project_env,
    read_env_value,
    resolve_long_pause_threshold,
)
from utterscope.diarization import DiarizationError, FixedLearnerSelector
from utterscope.llm import LlmError
from utterscope.models import AnalyzeRequest, AnalyzeResult
from utterscope.pipeline import run as run_pipeline
from utterscope.runtime import enable_quiet_mode
from utterscope.vad import VadError

# Load project .env early so backends can read HF_TOKEN and similar secrets.
load_project_env()

app = typer.Typer(
    name="utterscope",
    help="Local-first speaking analysis for language learners.",
    no_args_is_help=False,
    invoke_without_command=True,
)

EXIT_USER_ERROR = 1
EXIT_RUNTIME_ERROR = 2

_PIPELINE_ERRORS = (
    AudioPreparationError,
    VadError,
    DiarizationError,
    AsrError,
    LlmError,
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
    """UtterScope CLI.

    Run with no subcommand to start the interactive analyze flow.
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    if not verbose:
        enable_quiet_mode()

    if ctx.invoked_subcommand is not None:
        return

    try:
        result = run_interactive(verbose=verbose)
    except (typer.Abort, KeyboardInterrupt) as exc:
        console.print("\n[dim]Cancelled.[/dim]")
        raise typer.Exit(EXIT_USER_ERROR) from exc
    except _PIPELINE_ERRORS as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(EXIT_RUNTIME_ERROR) from exc
    except OSError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(EXIT_USER_ERROR) from exc
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(EXIT_RUNTIME_ERROR) from exc

    _print_result(result)


@app.command()
def setup() -> None:
    """Interactively save local settings to ``.env``."""
    run_setup()


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
    ] = DEFAULT_ASR_MODEL,
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
    long_pause_threshold: Annotated[
        float | None,
        typer.Option(
            "--long-pause-threshold",
            help=(
                "Pauses at or above this many seconds count as long. "
                "Overrides .env / setup (default: 1.0)."
            ),
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Directory for analysis outputs."),
    ] = None,
) -> None:
    """Analyze a recorded lesson or conversation (non-interactive flags)."""
    try:
        threshold = resolve_long_pause_threshold(long_pause_threshold)
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(EXIT_USER_ERROR) from exc

    if llm and not read_env_value(GEMINI_API_KEY):
        console.print(
            f"[red]Error:[/red] {GEMINI_API_KEY} is not set. "
            "Run `utterscope setup` or export the key, "
            "or pass --no-llm to skip feedback."
        )
        raise typer.Exit(EXIT_USER_ERROR)

    output_dir = (output or Path.cwd()).resolve()
    request = AnalyzeRequest(
        audio_path=audio,
        model=model,
        output_dir=output_dir,
        llm=llm,
        learner_speaker=learner,
        long_pause_threshold_seconds=threshold,
    )
    verbose = bool(ctx.obj.get("verbose", False))

    console.print(
        Panel.fit(
            f"[bold]{audio.name}[/bold]",
            title="UtterScope",
            border_style="cyan",
        )
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

    _print_result(result)


def _print_result(result: AnalyzeResult) -> None:
    console.print()
    console.print(f"Transcript → {result.transcript_path}")
    if result.analysis_path is not None:
        console.print(f"Analysis → {result.analysis_path}")
    if result.feedback_path is not None:
        console.print(f"Feedback → {result.feedback_path}")
    if result.learner_speaker is not None:
        console.print(f"Learner → {result.learner_speaker}")
    if result.analysis_document is not None:
        metrics = result.analysis_document.metrics
        console.print(
            f"Metrics → {metrics.speaking_time_seconds:.1f}s speaking, "
            f"{metrics.speaking_ratio:.0%} ratio, "
            f"{metrics.wpm:.0f} WPM, "
            f"{metrics.filler_count} fillers"
        )
    if result.feedback_document is not None:
        feedback = result.feedback_document
        console.print(
            f"LLM → {len(feedback.issues)} issues, "
            f"{len(feedback.recurring_patterns)} patterns"
        )
