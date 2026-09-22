"""Chronological dialogue turns derived from transcript segments."""

from __future__ import annotations

from pydantic import BaseModel, Field

from utterscope.models import Transcript

# Gaps at or below this are treated as continuous speech within a turn
# for LLM Pass B and speaking metrics.
TURN_GAP_SECONDS = 0.05


class DialogueTurn(BaseModel):
    """One contiguous stretch of speech by a single speaker."""

    index: int = Field(ge=0)
    speaker_id: str
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    text: str
    role: str | None = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def build_dialogue_turns(
    transcript: Transcript,
    *,
    learner_speaker: str | None = None,
    max_gap_seconds: float = TURN_GAP_SECONDS,
) -> list[DialogueTurn]:
    """Merge consecutive same-speaker segments into dialogue turns.

    Segments from the same speaker are merged when the gap between them is at
    most ``max_gap_seconds``. LLM feedback keeps the tight default
    (``TURN_GAP_SECONDS``); report display uses a larger gap so consecutive
    speech reads as one turn.
    """
    if max_gap_seconds < 0:
        msg = "max_gap_seconds must be >= 0"
        raise ValueError(msg)

    ordered = sorted(
        (segment for segment in transcript.segments if segment.speaker is not None),
        key=lambda segment: (segment.start, segment.end),
    )
    merged: list[tuple[str, float, float, str]] = []
    for segment in ordered:
        speaker_id = segment.speaker
        assert speaker_id is not None
        text = segment.text.strip()
        if merged:
            prev_speaker, prev_start, prev_end, prev_text = merged[-1]
            gap = segment.start - prev_end
            if prev_speaker == speaker_id and gap <= max_gap_seconds:
                merged_text = f"{prev_text} {text}".strip() if text else prev_text
                merged[-1] = (
                    speaker_id,
                    prev_start,
                    max(prev_end, segment.end),
                    merged_text,
                )
                continue
        merged.append((speaker_id, segment.start, segment.end, text))

    turns: list[DialogueTurn] = []
    for index, (speaker_id, start, end, text) in enumerate(merged):
        role = None
        if learner_speaker is not None:
            role = "student" if speaker_id == learner_speaker else "other"
        turns.append(
            DialogueTurn(
                index=index,
                speaker_id=speaker_id,
                start=start,
                end=end,
                text=text,
                role=role,
            )
        )
    return turns


def learner_turn_indices(
    turns: list[DialogueTurn],
    *,
    learner_speaker: str,
) -> list[int]:
    """Return turn indices belonging to the learner."""
    return [
        turn.index
        for turn in turns
        if turn.speaker_id == learner_speaker and turn.text.strip()
    ]


def context_window(
    turns: list[DialogueTurn],
    *,
    center_index: int,
    radius: int,
) -> list[DialogueTurn]:
    """Return turns in ``[center - radius, center + radius]`` inclusive."""
    if radius < 0:
        msg = "radius must be >= 0"
        raise ValueError(msg)
    start = max(0, center_index - radius)
    end = min(len(turns), center_index + radius + 1)
    return turns[start:end]
