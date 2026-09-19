"""Interactive local setup command."""

from __future__ import annotations

from pathlib import Path

import typer

from utterscope.cli.console import console
from utterscope.cli.select import RadioOption, select_radio
from utterscope.config.env import (
    DEFAULT_OUTPUT_ROOT_NAME,
    GEMINI_API_KEY,
    HF_TOKEN_KEY,
    LONG_PAUSE_THRESHOLD_KEY,
    OUTPUT_ASK_KEY,
    OUTPUT_ROOT_KEY,
    is_output_ask_enabled,
    mask_secret,
    project_env_path,
    read_env_value,
    resolve_long_pause_threshold,
    upsert_env_value,
)
from utterscope.models import DEFAULT_LONG_PAUSE_THRESHOLD_SECONDS

HF_TOKEN_HELP_URL = "https://huggingface.co/settings/tokens"
HF_MODEL_URL = "https://huggingface.co/pyannote/speaker-diarization-community-1"
GEMINI_KEY_HELP_URL = "https://aistudio.google.com/apikey"


def run_setup(*, project_dir: Path | None = None) -> Path:
    """Interactively configure local settings and write them to ``.env``."""
    directory = project_dir or Path.cwd()
    env_path = project_env_path(directory)

    console.print()
    console.print("[bold]UtterScope setup[/bold]")
    console.print(f"Settings file: {env_path}")
    console.print()

    while True:
        _print_current_settings(directory)
        choice = select_radio(
            title="What would you like to configure?",
            options=[
                RadioOption(
                    label="Hugging Face token",
                    detail="required for speaker diarization",
                ),
                RadioOption(
                    label="Gemini API key",
                    detail="required for --llm feedback",
                ),
                RadioOption(
                    label="Long pause threshold",
                    detail="seconds; used by analyze metrics",
                ),
                RadioOption(
                    label="Output directory",
                    detail="analysis result root / ask every time",
                ),
                RadioOption(
                    label="Done",
                    detail="finish setup",
                ),
            ],
        )
        console.print()
        if choice == 0:
            _configure_hf_token(directory)
        elif choice == 1:
            _configure_gemini_api_key(directory)
        elif choice == 2:
            _configure_long_pause_threshold(directory)
        elif choice == 3:
            _configure_output_directory(directory)
        else:
            console.print("[dim]Setup finished.[/dim]")
            break
        console.print()

    return env_path


def _print_current_settings(directory: Path) -> None:
    token = read_env_value(HF_TOKEN_KEY, directory)
    gemini = read_env_value(GEMINI_API_KEY, directory)
    threshold = resolve_long_pause_threshold(project_dir=directory)
    stored = read_env_value(LONG_PAUSE_THRESHOLD_KEY, directory)
    ask = is_output_ask_enabled(directory)
    configured_root = read_env_value(OUTPUT_ROOT_KEY, directory)

    console.print("[bold]Current settings[/bold]")
    if token:
        console.print(f"  HF_TOKEN: [dim]{mask_secret(token)}[/dim]")
    else:
        console.print("  HF_TOKEN: [dim](not set)[/dim]")
    if gemini:
        console.print(f"  GEMINI_API_KEY: [dim]{mask_secret(gemini)}[/dim]")
    else:
        console.print("  GEMINI_API_KEY: [dim](not set)[/dim]")
    source = "from .env" if stored is not None else "default"
    console.print(
        f"  Long pause threshold: [cyan]{threshold:g}s[/cyan]  [dim]({source})[/dim]"
    )
    if ask:
        console.print(
            "  Output root: [cyan]ask every time[/cyan]  [dim](interactive only)[/dim]"
        )
    elif configured_root:
        console.print(f"  Output root: [cyan]{configured_root}[/cyan]")
    else:
        console.print(
            f"  Output root: [cyan]./{DEFAULT_OUTPUT_ROOT_NAME}[/cyan]  "
            "[dim](default)[/dim]"
        )
    console.print()


def _configure_hf_token(directory: Path) -> None:
    console.print("Speaker diarization needs a Hugging Face token for pyannote models.")
    console.print(f"  1. Create a token: {HF_TOKEN_HELP_URL}")
    console.print(f"  2. Accept model terms: {HF_MODEL_URL}")
    console.print()

    current = read_env_value(HF_TOKEN_KEY, directory)
    if current:
        choice = select_radio(
            title="HF_TOKEN is already set. Update it?",
            options=[
                RadioOption(label="Update token", detail="overwrite .env"),
                RadioOption(label="Keep current", detail="leave unchanged"),
            ],
        )
        if choice == 1:
            console.print("[dim]Kept existing HF_TOKEN.[/dim]")
            return

    token = typer.prompt("HF_TOKEN", hide_input=True).strip()
    if not token:
        console.print("[red]Empty token. Nothing was saved.[/red]")
        return

    written = upsert_env_value(HF_TOKEN_KEY, token, directory)
    console.print(
        f"[green]✔[/green] Saved {HF_TOKEN_KEY} to {written}  "
        f"[dim]({mask_secret(token)})[/dim]"
    )


def _configure_gemini_api_key(directory: Path) -> None:
    console.print("LLM feedback (--llm) needs a Gemini API key from Google AI Studio.")
    console.print(f"  Create a key: {GEMINI_KEY_HELP_URL}")
    console.print()

    current = read_env_value(GEMINI_API_KEY, directory)
    if current:
        choice = select_radio(
            title="GEMINI_API_KEY is already set. Update it?",
            options=[
                RadioOption(label="Update key", detail="overwrite .env"),
                RadioOption(label="Keep current", detail="leave unchanged"),
            ],
        )
        if choice == 1:
            console.print("[dim]Kept existing GEMINI_API_KEY.[/dim]")
            return

    key = typer.prompt("GEMINI_API_KEY", hide_input=True).strip()
    if not key:
        console.print("[red]Empty key. Nothing was saved.[/red]")
        return

    written = upsert_env_value(GEMINI_API_KEY, key, directory)
    console.print(
        f"[green]✔[/green] Saved {GEMINI_API_KEY} to {written}  "
        f"[dim]({mask_secret(key)})[/dim]"
    )


def _configure_long_pause_threshold(directory: Path) -> None:
    current = resolve_long_pause_threshold(project_dir=directory)
    console.print(
        "Pauses at or above this many seconds count as long pauses in analysis.json."
    )
    console.print(
        f"Default is {DEFAULT_LONG_PAUSE_THRESHOLD_SECONDS:g}s. "
        "CLI --long-pause-threshold overrides this setting."
    )
    console.print()

    raw = typer.prompt(
        "Long pause threshold (seconds)",
        default=str(current),
    ).strip()
    try:
        value = float(raw)
    except ValueError:
        console.print("[red]Invalid number. Nothing was saved.[/red]")
        return
    if value <= 0:
        console.print("[red]Threshold must be greater than 0.[/red]")
        return

    stored = f"{value:g}"
    written = upsert_env_value(LONG_PAUSE_THRESHOLD_KEY, stored, directory)
    console.print(
        f"[green]✔[/green] Saved {LONG_PAUSE_THRESHOLD_KEY}={stored} to {written}"
    )


def _configure_output_directory(directory: Path) -> None:
    console.print(
        "Each analyze run creates a folder under the output root:\n"
        "  YYMMDD-n_<audio-stem>/ with JSON artifacts and a copy of the audio."
    )
    console.print(
        f"Default root is [cyan]./{DEFAULT_OUTPUT_ROOT_NAME}[/cyan] "
        "under the current working directory."
    )
    console.print(
        "[dim]Ask every time applies to interactive mode only; "
        "non-interactive analyze requires -o or a fixed root.[/dim]"
    )
    console.print()

    choice = select_radio(
        title="Output directory setting",
        options=[
            RadioOption(
                label="Use default",
                detail=f"./{DEFAULT_OUTPUT_ROOT_NAME} under cwd",
            ),
            RadioOption(
                label="Set absolute path",
                detail="save UTTERSCOPE_OUTPUT_ROOT",
            ),
            RadioOption(
                label="Ask every time",
                detail="interactive mode only",
            ),
        ],
    )
    if choice == 0:
        upsert_env_value(OUTPUT_ASK_KEY, "0", directory)
        env_path = upsert_env_value(OUTPUT_ROOT_KEY, "", directory)
        console.print(
            f"[green]✔[/green] Using default [cyan]./{DEFAULT_OUTPUT_ROOT_NAME}"
            f"[/cyan] ({env_path})"
        )
        return

    if choice == 2:
        written = upsert_env_value(OUTPUT_ASK_KEY, "1", directory)
        console.print(
            f"[green]✔[/green] Saved {OUTPUT_ASK_KEY}=1 to {written} (interactive only)"
        )
        return

    current = read_env_value(OUTPUT_ROOT_KEY, directory)
    default = current or str(Path.cwd() / DEFAULT_OUTPUT_ROOT_NAME)
    raw = typer.prompt("Absolute output root", default=default).strip()
    path = Path(raw).expanduser()
    if not path.is_absolute():
        console.print("[red]Path must be absolute. Nothing was saved.[/red]")
        return
    upsert_env_value(OUTPUT_ASK_KEY, "0", directory)
    written = upsert_env_value(OUTPUT_ROOT_KEY, str(path.resolve()), directory)
    console.print(
        f"[green]✔[/green] Saved {OUTPUT_ROOT_KEY}={path.resolve()} to {written}"
    )
