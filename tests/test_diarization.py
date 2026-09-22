"""Tests for learner speaker selection helpers."""

from __future__ import annotations

from utterscope.diarization import build_speaker_previews
from utterscope.models import Segment, Transcript


def test_build_speaker_previews() -> None:
    transcript = Transcript(
        segments=[
            Segment(start=0.0, end=1.0, text="A", speaker="SPEAKER_00"),
            Segment(start=1.0, end=3.0, text="B", speaker="SPEAKER_01"),
            Segment(start=3.0, end=4.0, text="C", speaker="SPEAKER_01"),
        ],
        full_text="A B C",
    )

    previews = build_speaker_previews(transcript)

    assert len(previews) == 2
    by_id = {preview.speaker_id: preview for preview in previews}
    assert by_id["SPEAKER_00"].speaking_time_seconds == 1.0
    assert by_id["SPEAKER_01"].speaking_time_seconds == 3.0
    assert by_id["SPEAKER_01"].sample_texts == ["B", "C"]
