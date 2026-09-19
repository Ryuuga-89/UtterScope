"""Internal diarization result types."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class SpeakerTurn(BaseModel):
    """A contiguous region attributed to one speaker."""

    start: float = Field(ge=0)
    end: float = Field(ge=0)
    speaker_id: str

    @model_validator(mode="after")
    def validate_time_range(self) -> SpeakerTurn:
        if self.end < self.start:
            msg = "end must be greater than or equal to start"
            raise ValueError(msg)
        return self


class DiarizationResult(BaseModel):
    """Speaker turns produced by a diarization backend."""

    turns: list[SpeakerTurn] = Field(default_factory=list)

    def speaker_ids(self) -> list[str]:
        """Return unique speaker ids in first-seen order."""
        seen: list[str] = []
        for turn in self.turns:
            if turn.speaker_id not in seen:
                seen.append(turn.speaker_id)
        return seen
