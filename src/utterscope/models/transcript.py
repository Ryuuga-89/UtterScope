"""Transcript and JSON document models."""

from __future__ import annotations

from pydantic import BaseModel, Field

from utterscope.models.segment import Segment

SCHEMA_VERSION = 1


class Transcript(BaseModel):
    """Timestamped transcript built from utterance chunks."""

    language: str | None = None
    segments: list[Segment] = Field(default_factory=list)
    full_text: str = ""


class TranscriptDocument(BaseModel):
    """Serializable transcript artifact written to transcript.json."""

    schema_version: int = SCHEMA_VERSION
    source_audio: str
    model: str
    transcript: Transcript
