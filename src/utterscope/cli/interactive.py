"""Interactive analyze session (``utterscope`` with no subcommand)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

from utterscope.asr.models import (
    ASR_MODEL_CHOICES,
    DEFAULT_ASR_MODEL,
    list_asr_models,
)
from utterscope.cli.console import ANSI_YELLOW, ansi_rgb, console
from utterscope.cli.history_record import record_history
from utterscope.cli.learner import InteractiveLearnerSelector
from utterscope.cli.progress import CliProgress
from utterscope.cli.select import RadioOption, select_radio
from utterscope.config import (
    GEMINI_API_KEY,
    read_env_value,
    resolve_long_pause_threshold,
    resolve_output_root,
)
from utterscope.llm import LlmError, generate_feedback_document
from utterscope.llm.catalog import (
    LLM_PROVIDERS,
    default_gemini_model_index,
    models_for_provider,
)
from utterscope.models import (
    DEFAULT_LLM_PROVIDER,
    AnalyzeRequest,
    AnalyzeResult,
)
from utterscope.output import create_run_layout
from utterscope.pipeline import FEEDBACK_FILENAME
from utterscope.pipeline import run as run_pipeline
from utterscope.pipeline.analyze import PipelineProgress
from utterscope.report import write_feedback_document, write_reports

_AUDIO_SUFFIXES = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".mp4"}

# Blue (lighter / faster) → red (heavier / more accurate).
_ASR_GRADIENT_START = (64, 140, 255)
_ASR_GRADIENT_END = (235, 64, 64)


@dataclass(frozen=True)
class InteractiveLlmChoice:
    """LLM settings chosen after the core analyze pass."""

    enabled: bool
    provider: str | None = None
    model: str | None = None


def prompt_audio_path() -> Path:
    """Ask for an absolute audio path and validate it."""
    console.print()
    body = Text()
    body.append("Audio file\n", style="bold")
    body.append(
        "Paste or type an absolute path to the lesson recording.\n",
        style="dim",
    )
    body.append("Tip: ", style="dim")
    body.append("drag a file into the terminal", style="cyan")
    body.append(" to insert its path.\n", style="dim")
    body.append("Formats: ", style="dim")
    body.append(", ".join(sorted(_AUDIO_SUFFIXES)), style="dim cyan")
    console.print(
        Panel(
            body,
            title="[bold cyan]1 · Input[/bold cyan]",
            border_style="cyan",
            padding=(1, 2),
        )
    )

    while True:
        console.print()
        raw = (
            Prompt.ask(
                "[bold cyan]❯[/bold cyan] Absolute path",
                console=console,
            )
            .strip()
            .strip("'\"")
        )
        if not raw:
            console.print("  [red]✖[/red] Path is required.")
            continue
        path = Path(raw).expanduser()
        if not path.is_absolute():
            console.print(
                "  [red]✖[/red] Use an absolute path (starts with [bold]/[/bold])."
            )
            continue
        if not path.is_file():
            console.print(f"  [red]✖[/red] File not found: [dim]{path}[/dim]")
            continue
        if path.suffix.lower() and path.suffix.lower() not in _AUDIO_SUFFIXES:
            console.print(
                f"  [yellow]![/yellow] Unusual extension "
                f"[bold]{path.suffix}[/bold] — continuing anyway."
            )
        console.print(f"  [green]✔[/green] [bold]{path.resolve()}[/bold]")
        return path.resolve()


def prompt_asr_model() -> str:
    """Ask which Whisper / mlx-whisper model to use."""
    models = list_asr_models()
    count = max(1, len(models) - 1)
    options: list[RadioOption] = []
    for index, model in enumerate(models):
        t = index / count
        color = ansi_rgb(*_lerp_rgb(_ASR_GRADIENT_START, _ASR_GRADIENT_END, t))
        detail = "downloaded" if model.downloaded else ""
        options.append(
            RadioOption(
                label=model.name,
                detail=detail,
                label_ansi=color,
                detail_ansi=ANSI_YELLOW if model.downloaded else "",
            )
        )
    default_index = 0
    if DEFAULT_ASR_MODEL in ASR_MODEL_CHOICES:
        default_index = ASR_MODEL_CHOICES.index(DEFAULT_ASR_MODEL)
    console.print()
    index = select_radio(
        title="ASR model",
        subtitle="blue = lighter/faster  ·  red = higher accuracy",
        options=options,
        default_index=default_index,
    )
    return models[index].name


def _lerp_rgb(
    start: tuple[int, int, int],
    end: tuple[int, int, int],
    t: float,
) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, t))
    return (
        int(start[0] + (end[0] - start[0]) * t),
        int(start[1] + (end[1] - start[1]) * t),
        int(start[2] + (end[2] - start[2]) * t),
    )


def prompt_llm_choice() -> InteractiveLlmChoice:
    """Ask whether to run LLM feedback, then provider + model."""
    console.print()
    provider_options = [
        RadioOption(label="Skip LLM analysis", detail="metrics only"),
        *[
            RadioOption(label=provider.label, detail=provider.detail)
            for provider in LLM_PROVIDERS
        ],
    ]
    provider_index = select_radio(
        title="LLM provider",
        options=provider_options,
        default_index=1 if LLM_PROVIDERS else 0,
    )
    if provider_index == 0:
        return InteractiveLlmChoice(enabled=False)

    provider = LLM_PROVIDERS[provider_index - 1]
    models = models_for_provider(provider.id)
    model_options = [
        RadioOption(
            label=model.id,
            detail=f"{model.family}" + (f" · {model.detail}" if model.detail else ""),
        )
        for model in models
    ]
    console.print()
    model_index = select_radio(
        title=f"LLM model ({provider.label})",
        options=model_options,
        default_index=default_gemini_model_index(),
    )
    return InteractiveLlmChoice(
        enabled=True,
        provider=provider.id,
        model=models[model_index].id,
    )


def prompt_output_root_absolute() -> str:
    """Ask for an absolute output root path (interactive ask mode)."""
    console.print()
    console.print(
        "[bold]Output root[/bold]\n"
        "[dim]Directory that will contain timestamped run folders.[/dim]"
    )
    while True:
        raw = (
            Prompt.ask(
                "[bold cyan]❯[/bold cyan] Absolute output root",
                console=console,
            )
            .strip()
            .strip("'\"")
        )
        if not raw:
            console.print("  [red]✖[/red] Path is required.")
            continue
        path = Path(raw).expanduser()
        if not path.is_absolute():
            console.print(
                "  [red]✖[/red] Use an absolute path (starts with [bold]/[/bold])."
            )
            continue
        console.print(f"  [green]✔[/green] [bold]{path.resolve()}[/bold]")
        return str(path.resolve())


def run_interactive(
    *,
    verbose: bool = False,
    output_root: Path | None = None,
) -> AnalyzeResult:
    """Run the guided analyze flow and return the final result."""
    audio = prompt_audio_path()
    asr_model = prompt_asr_model()
    threshold = resolve_long_pause_threshold(None)
    root = resolve_output_root(
        output_root,
        interactive=True,
        prompt_absolute_path=prompt_output_root_absolute,
    )
    layout = create_run_layout(root, audio)

    console.print()
    console.print(
        Panel.fit(
            f"[bold]{audio}[/bold]\n"
            f"[dim]ASR: {asr_model}[/dim]\n"
            f"[dim]Run: {layout.run_dir}[/dim]",
            title="UtterScope",
            border_style="cyan",
        )
    )

    request = AnalyzeRequest(
        audio_path=audio,
        model=asr_model,
        output_dir=layout.run_dir,
        llm=False,
        long_pause_threshold_seconds=threshold,
    )
    progress = CliProgress()
    try:
        result = run_pipeline(
            request,
            progress=progress,
            learner_selector=InteractiveLearnerSelector(),
            quiet=not verbose,
        )

        llm_choice = prompt_llm_choice()
        if not llm_choice.enabled:
            _record_and_return(result, progress)
            return result

        assert llm_choice.provider is not None
        assert llm_choice.model is not None
        if llm_choice.provider != DEFAULT_LLM_PROVIDER:
            msg = f"unsupported LLM provider: {llm_choice.provider}"
            raise LlmError(msg)
        if not read_env_value(GEMINI_API_KEY):
            msg = (
                f"{GEMINI_API_KEY} is not set. "
                "Run `utterscope setup` or export the key, "
                "or choose Skip LLM analysis."
            )
            raise LlmError(msg)

        result = _run_feedback(
            result,
            llm_model=llm_choice.model,
            progress=progress,
        )
        _record_and_return(result, progress)
        return result
    except BaseException:
        progress.cancel()
        raise


def _record_and_return(
    result: AnalyzeResult,
    progress: PipelineProgress | None,
) -> None:
    record_history(result, progress=progress)


def _run_feedback(
    result: AnalyzeResult,
    *,
    llm_model: str,
    progress: PipelineProgress | None,
) -> AnalyzeResult:
    if result.learner_speaker is None:
        msg = "learner speaker is required before LLM feedback"
        raise LlmError(msg)

    if progress is not None:
        progress.begin("generate feedback")

    def on_progress(label: str) -> None:
        if progress is not None:
            progress.update(label)

    feedback_document = generate_feedback_document(
        result.document.transcript,
        learner_speaker=result.learner_speaker,
        source_audio=result.document.source_audio,
        asr_model=result.document.model,
        llm_model=llm_model,
        on_progress=on_progress,
    )
    if progress is not None:
        progress.end(
            "generate feedback",
            f"{len(feedback_document.issues)} issues",
        )

    feedback_path = result.transcript_path.parent / FEEDBACK_FILENAME
    if progress is not None:
        progress.begin("write feedback")
    written = write_feedback_document(feedback_document, feedback_path)
    if progress is not None:
        progress.end("write feedback")

    if result.analysis_document is None:
        msg = "analysis document is required before regenerating reports"
        raise LlmError(msg)

    if progress is not None:
        progress.begin("generate reports")
    audio_filename = (
        result.source_audio_path.name
        if result.source_audio_path is not None
        else result.document.source_audio
    )
    report_md_path, report_html_path = write_reports(
        result.transcript_path.parent,
        transcript_document=result.document,
        analysis_document=result.analysis_document,
        feedback_document=feedback_document,
        duration_seconds=result.duration_seconds,
        audio_filename=audio_filename,
    )
    if progress is not None:
        progress.end("generate reports", "report.md, report.html")

    return result.model_copy(
        update={
            "feedback_document": feedback_document,
            "feedback_path": written,
            "report_md_path": report_md_path,
            "report_html_path": report_html_path,
        }
    )
