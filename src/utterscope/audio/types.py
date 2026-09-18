"""Internal prepared-audio representation (not part of the public JSON schema)."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class PreparedAudio(BaseModel):
    """Audio normalized for downstream ASR."""

    path: Path
    sample_rate: int = Field(gt=0)
    channels: int = Field(gt=0)
    duration_seconds: float = Field(ge=0)
