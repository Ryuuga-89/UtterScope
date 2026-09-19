"""Assign diarization labels onto transcript segments."""

from __future__ import annotations

from utterscope.diarization.types import DiarizationResult, SpeakerTurn
from utterscope.models import Segment, Transcript


def assign_speakers(
    transcript: Transcript,
    diarization: DiarizationResult,
) -> Transcript:
    """Copy transcript segments with the best-overlapping speaker label."""
    segments = [
        segment.model_copy(
            update={
                "speaker": _best_speaker(segment, diarization.turns),
            }
        )
        for segment in transcript.segments
    ]
    return transcript.model_copy(update={"segments": segments})


def _best_speaker(
    segment: Segment,
    turns: list[SpeakerTurn],
) -> str | None:
    best_id: str | None = None
    best_overlap = 0.0
    for turn in turns:
        overlap = _overlap_seconds(
            segment.start,
            segment.end,
            turn.start,
            turn.end,
        )
        if overlap > best_overlap:
            best_overlap = overlap
            best_id = turn.speaker_id
    return best_id


def _overlap_seconds(
    start_a: float,
    end_a: float,
    start_b: float,
    end_b: float,
) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))
