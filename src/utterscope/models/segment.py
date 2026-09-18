"""Transcript segment with timestamps in seconds."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class Segment(BaseModel):
    """A single utterance chunk with start/end times in seconds."""

    start: float = Field(ge=0, description="Segment start time in seconds.")
    end: float = Field(ge=0, description="Segment end time in seconds.")
    text: str

    @model_validator(mode="after")
    def validate_time_range(self) -> Segment:
        if self.end < self.start:
            msg = "end must be greater than or equal to start"
            raise ValueError(msg)
        return self
