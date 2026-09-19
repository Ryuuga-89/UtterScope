"""Tests for deterministic speaking metrics."""

from __future__ import annotations

from utterscope.metrics import compute_speaking_metrics, count_fillers
from utterscope.models import Segment, Transcript


def test_count_fillers_words_and_phrases() -> None:
    text = "Um, I mean, uh, you know, like, hello"
    assert count_fillers(text) == 5


def test_compute_speaking_metrics_for_learner() -> None:
    transcript = Transcript(
        language="en",
        segments=[
            Segment(
                start=0.0,
                end=2.0,
                text="Hello there",
                speaker="SPEAKER_00",
            ),
            Segment(
                start=2.5,
                end=4.5,
                text="Um yes I think so",
                speaker="SPEAKER_01",
            ),
            Segment(
                start=6.0,
                end=7.0,
                text="Okay",
                speaker="SPEAKER_01",
            ),
            Segment(
                start=7.0,
                end=9.0,
                text="Right",
                speaker="SPEAKER_00",
            ),
        ],
        full_text="Hello there Um yes I think so Okay Right",
    )

    metrics = compute_speaking_metrics(
        transcript,
        learner_speaker="SPEAKER_01",
        long_pause_threshold_seconds=1.0,
    )

    # SPEAKER_01: 2.0s + 1.0s = 3.0s; total speaking 7.0s
    assert metrics.speaking_time_seconds == 3.0
    assert metrics.speaking_ratio == round(3.0 / 7.0, 4)
    assert metrics.turn_count == 2
    assert metrics.average_turn_seconds == 1.5
    assert metrics.pause_count == 1
    assert metrics.long_pause_count == 1
    assert metrics.filler_count == 1
    # um yes i think so okay → 6 words over 3.0s
    assert metrics.wpm == 120.0
