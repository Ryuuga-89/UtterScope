"""Pipeline request and result models."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from utterscope.models.analysis import (
    DEFAULT_LONG_PAUSE_THRESHOLD_SECONDS,
    AnalysisDocument,
)
from utterscope.models.feedback import (
    DEFAULT_FEEDBACK_CONTEXT_TURNS,
    DEFAULT_LLM_MODEL,
    FeedbackDocument,
)
from utterscope.models.transcript import TranscriptDocument


class AnalyzeRequest(BaseModel):
    """Inputs for a single analyze run."""

    audio_path: Path
    model: str
    output_dir: Path
    llm: bool = Field(
        default=True,
        description="When true, run Gemini feedback and write feedback.json.",
    )
    learner_speaker: str | None = Field(
        default=None,
        description="Selected learner speaker id, when already known.",
    )
    long_pause_threshold_seconds: float = Field(
        default=DEFAULT_LONG_PAUSE_THRESHOLD_SECONDS,
        gt=0,
        description="Pauses at or above this many seconds count as long.",
    )
    llm_model: str = Field(
        default=DEFAULT_LLM_MODEL,
        description="Gemini model id used when llm=True.",
    )
    feedback_context_turns: int = Field(
        default=DEFAULT_FEEDBACK_CONTEXT_TURNS,
        ge=0,
        description=(
            "Neighboring turns (±N) passed with each learner turn in Pass B."
        ),
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
    learner_speaker: str | None = None
    analysis_document: AnalysisDocument | None = None
    analysis_path: Path | None = None
    feedback_document: FeedbackDocument | None = None
    feedback_path: Path | None = None
