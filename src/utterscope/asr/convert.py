"""Convert Whisper-style result dicts into public Transcript models."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from utterscope.models import Segment, Transcript


def transcript_from_whisper_result(result: Mapping[str, Any]) -> Transcript:
    """Build a Transcript from a whispermlx / Whisper-style result dict.

    When segments include ``speaker`` (or word-level speakers), those labels
    are preserved on each ``Segment``.
    """
    raw_segments = result.get("segments") or []
    segments: list[Segment] = []
    for item in raw_segments:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        segments.append(
            Segment(
                start=float(item["start"]),
                end=float(item["end"]),
                text=text,
                speaker=_speaker_of(item),
            )
        )

    full_text = str(result.get("text", "")).strip()
    if not full_text and segments:
        full_text = " ".join(segment.text for segment in segments)

    language = result.get("language")
    language_value = str(language) if language is not None else None

    return Transcript(
        language=language_value,
        segments=segments,
        full_text=full_text,
    )


def _speaker_of(segment: dict[str, Any]) -> str | None:
    for key in ("speaker", "Speaker"):
        value = segment.get(key)
        if value is not None and str(value).strip():
            return str(value)
    words = segment.get("words") or []
    speakers = [
        str(word.get("speaker"))
        for word in words
        if isinstance(word, dict) and word.get("speaker")
    ]
    if not speakers:
        return None
    return max(set(speakers), key=speakers.count)
