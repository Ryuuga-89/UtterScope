"""Internal VAD result types (not part of the public JSON schema)."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class SpeechInterval(BaseModel):
    """A contiguous speech region in seconds."""

    start: float = Field(ge=0)
    end: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_time_range(self) -> SpeechInterval:
        if self.end < self.start:
            msg = "end must be greater than or equal to start"
            raise ValueError(msg)
        return self


class VadResult(BaseModel):
    """Speech intervals detected by a VAD backend."""

    intervals: list[SpeechInterval] = Field(default_factory=list)
