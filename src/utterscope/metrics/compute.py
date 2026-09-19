"""Deterministic speaking metrics from a diarized transcript."""

from __future__ import annotations

from dataclasses import dataclass

from utterscope.metrics.fillers import count_fillers, tokenize_words
from utterscope.models import (
    DEFAULT_LONG_PAUSE_THRESHOLD_SECONDS,
    Segment,
    SpeakerRole,
    SpeakingMetrics,
    Transcript,
)

# Gaps at or below this are treated as continuous speech within a turn.
_TURN_GAP_SECONDS = 0.05


@dataclass(frozen=True)
class _Turn:
    speaker_id: str
    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def compute_speaking_metrics(
    transcript: Transcript,
    *,
    learner_speaker: str,
    long_pause_threshold_seconds: float = DEFAULT_LONG_PAUSE_THRESHOLD_SECONDS,
) -> SpeakingMetrics:
    """Compute learner-focused speaking metrics from ``transcript``."""
    turns = _build_turns(transcript.segments)
    learner_turns = [turn for turn in turns if turn.speaker_id == learner_speaker]

    speaking_time = sum(turn.duration for turn in learner_turns)
    total_speaking = sum(turn.duration for turn in turns)
    speaking_ratio = (
        speaking_time / total_speaking if total_speaking > 0 else 0.0
    )

    learner_text = " ".join(turn.text for turn in learner_turns)
    word_count = len(tokenize_words(learner_text))
    wpm = (word_count / speaking_time * 60.0) if speaking_time > 0 else 0.0

    turn_count = len(learner_turns)
    average_turn = speaking_time / turn_count if turn_count > 0 else 0.0

    pause_count = 0
    long_pause_count = 0
    for previous, current in zip(learner_turns, learner_turns[1:], strict=False):
        gap = current.start - previous.end
        if gap <= 0:
            continue
        pause_count += 1
        if gap >= long_pause_threshold_seconds:
            long_pause_count += 1

    return SpeakingMetrics(
        speaking_time_seconds=round(speaking_time, 3),
        speaking_ratio=round(speaking_ratio, 4),
        wpm=round(wpm, 1),
        turn_count=turn_count,
        average_turn_seconds=round(average_turn, 3),
        pause_count=pause_count,
        long_pause_count=long_pause_count,
        filler_count=count_fillers(learner_text),
    )


def build_speaker_roles(
    transcript: Transcript,
    *,
    learner_speaker: str,
) -> list[SpeakerRole]:
    """Assign student/other roles for speakers present in ``transcript``."""
    speaker_ids = sorted(
        {
            segment.speaker
            for segment in transcript.segments
            if segment.speaker is not None
        }
    )
    if learner_speaker not in speaker_ids:
        speaker_ids.append(learner_speaker)
        speaker_ids.sort()
    return [
        SpeakerRole(
            speaker_id=speaker_id,
            role="student" if speaker_id == learner_speaker else "other",
        )
        for speaker_id in speaker_ids
    ]


def _build_turns(segments: list[Segment]) -> list[_Turn]:
    ordered = sorted(
        (segment for segment in segments if segment.speaker is not None),
        key=lambda segment: (segment.start, segment.end),
    )
    turns: list[_Turn] = []
    for segment in ordered:
        speaker_id = segment.speaker
        assert speaker_id is not None
        text = segment.text.strip()
        if turns:
            previous = turns[-1]
            gap = segment.start - previous.end
            if (
                previous.speaker_id == speaker_id
                and gap <= _TURN_GAP_SECONDS
            ):
                merged_text = (
                    f"{previous.text} {text}".strip() if text else previous.text
                )
                turns[-1] = _Turn(
                    speaker_id=speaker_id,
                    start=previous.start,
                    end=max(previous.end, segment.end),
                    text=merged_text,
                )
                continue
        turns.append(
            _Turn(
                speaker_id=speaker_id,
                start=segment.start,
                end=segment.end,
                text=text,
            )
        )
    return turns
