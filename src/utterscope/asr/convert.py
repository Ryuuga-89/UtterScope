"""Convert Whisper-style result dicts into public Transcript models."""

from __future__ import annotations

from typing import Any

from utterscope.models import Segment, Transcript


def transcript_from_whisper_result(result: dict[str, Any]) -> Transcript:
    """Build a Transcript from an mlx-whisper / openai-whisper result dict."""
    raw_segments = result.get("segments") or []
    segments: list[Segment] = []
    for item in raw_segments:
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        segments.append(
            Segment(
                start=float(item["start"]),
                end=float(item["end"]),
                text=text,
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
