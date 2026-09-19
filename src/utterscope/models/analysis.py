"""Analysis document models written to analysis.json."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ANALYSIS_SCHEMA_VERSION = 1
DEFAULT_LONG_PAUSE_THRESHOLD_SECONDS = 1.0

SpeakerRoleName = Literal["student", "other"]


class SpeakerRole(BaseModel):
    """Role assignment for a diarized speaker."""

    speaker_id: str
    role: SpeakerRoleName


class SpeakingMetrics(BaseModel):
    """Deterministic speaking metrics for the selected learner."""

    speaking_time_seconds: float = Field(ge=0)
    speaking_ratio: float = Field(
        ge=0,
        le=1,
        description="Learner speaking time divided by total speaking time.",
    )
    wpm: float = Field(ge=0)
    turn_count: int = Field(ge=0)
    average_turn_seconds: float = Field(ge=0)
    pause_count: int = Field(ge=0)
    long_pause_count: int = Field(ge=0)
    filler_count: int = Field(ge=0)


class AnalysisDocument(BaseModel):
    """Serializable analysis artifact written to analysis.json."""

    schema_version: int = ANALYSIS_SCHEMA_VERSION
    source_audio: str
    model: str
    learner_speaker: str
    speakers: list[SpeakerRole]
    metrics: SpeakingMetrics
    long_pause_threshold_seconds: float = Field(
        default=DEFAULT_LONG_PAUSE_THRESHOLD_SECONDS,
        gt=0,
        description="Threshold used to classify long pauses.",
    )
