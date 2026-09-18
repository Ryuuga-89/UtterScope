"""Pipeline request and result models."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from utterscope.models.transcript import TranscriptDocument


class AnalyzeRequest(BaseModel):
    """Inputs for a single analyze run."""

    audio_path: Path
    model: str
    output_dir: Path
    llm: bool = Field(
        default=True,
        description="Accepted for CLI compatibility; ignored in v0.1.",
    )


class AnalyzeResult(BaseModel):
    """Outputs produced by a successful analyze run."""

    document: TranscriptDocument
    transcript_path: Path
    duration_seconds: float | None = Field(
        default=None,
        ge=0,
        description="Source audio duration in seconds, when known.",
    )
