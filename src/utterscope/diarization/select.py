"""Learner speaker selection helpers."""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, Field

from utterscope.models import Transcript


class SpeakerPreview(BaseModel):
    """Summary used to help the user pick the learner speaker."""

    speaker_id: str
    speaking_time_seconds: float = Field(ge=0)
    sample_texts: list[str] = Field(default_factory=list)


class LearnerSelector(Protocol):
    """Chooses which diarized speaker is the learner."""

    def select_learner(self, previews: list[SpeakerPreview]) -> str:
        """Return the selected learner speaker id."""


def build_speaker_previews(
    transcript: Transcript,
    *,
    sample_limit: int = 3,
) -> list[SpeakerPreview]:
    """Build previews for speakers present on transcript segments."""
    speaking_time: dict[str, float] = {}
    samples: dict[str, list[str]] = {}

    for segment in transcript.segments:
        if segment.speaker is None:
            continue
        duration = max(0.0, segment.end - segment.start)
        speaking_time[segment.speaker] = (
            speaking_time.get(segment.speaker, 0.0) + duration
        )
        texts = samples.setdefault(segment.speaker, [])
        if len(texts) < sample_limit and segment.text:
            texts.append(segment.text)

    return [
        SpeakerPreview(
            speaker_id=speaker_id,
            speaking_time_seconds=speaking_time[speaker_id],
            sample_texts=samples.get(speaker_id, []),
        )
        for speaker_id in speaking_time
    ]


class FixedLearnerSelector:
    """Selector that always returns a predetermined speaker id."""

    def __init__(self, speaker_id: str) -> None:
        self._speaker_id = speaker_id

    def select_learner(self, previews: list[SpeakerPreview]) -> str:
        ids = {preview.speaker_id for preview in previews}
        if self._speaker_id not in ids and previews:
            msg = (
                f"learner speaker {self._speaker_id!r} was not found "
                f"among {sorted(ids)}"
            )
            raise ValueError(msg)
        return self._speaker_id
