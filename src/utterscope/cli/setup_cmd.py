"""Interactive local setup command."""

from __future__ import annotations

from pathlib import Path

import typer

from utterscope.cli.console import console
from utterscope.cli.select import RadioOption, select_radio
from utterscope.config.env import (
    HF_TOKEN_KEY,
    LONG_PAUSE_THRESHOLD_KEY,
    mask_secret,
    project_env_path,
    read_env_value,
    resolve_long_pause_threshold,
    upsert_env_value,
)
from utterscope.models import DEFAULT_LONG_PAUSE_THRESHOLD_SECONDS

HF_TOKEN_HELP_URL = "https://huggingface.co/settings/tokens"
HF_MODEL_URL = (
    "https://huggingface.co/pyannote/speaker-diarization-community-1"
)


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
                    label="Long pause threshold",
                    detail="seconds; used by analyze metrics",
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
            _configure_long_pause_threshold(directory)
        else:
            console.print("[dim]Setup finished.[/dim]")
            break
        console.print()

    return env_path


def _print_current_settings(directory: Path) -> None:
    token = read_env_value(HF_TOKEN_KEY, directory)
    threshold = resolve_long_pause_threshold(project_dir=directory)
    stored = read_env_value(LONG_PAUSE_THRESHOLD_KEY, directory)

    console.print("[bold]Current settings[/bold]")
    if token:
        console.print(f"  HF_TOKEN: [dim]{mask_secret(token)}[/dim]")
    else:
        console.print("  HF_TOKEN: [dim](not set)[/dim]")
    source = "from .env" if stored is not None else "default"
    console.print(
        f"  Long pause threshold: [cyan]{threshold:g}s[/cyan]  [dim]({source})[/dim]"
    )
    console.print()


def _configure_hf_token(directory: Path) -> None:
    console.print(
        "Speaker diarization needs a Hugging Face token for pyannote models."
    )
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


def _configure_long_pause_threshold(directory: Path) -> None:
    current = resolve_long_pause_threshold(project_dir=directory)
    console.print(
        "Pauses at or above this many seconds count as long pauses "
        "in analysis.json."
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

    # Normalize display (avoid 1.5000 noise) while keeping precision in file.
    stored = f"{value:g}"
    written = upsert_env_value(LONG_PAUSE_THRESHOLD_KEY, stored, directory)
    console.print(
        f"[green]✔[/green] Saved {LONG_PAUSE_THRESHOLD_KEY}={stored} "
        f"to {written}"
    )
