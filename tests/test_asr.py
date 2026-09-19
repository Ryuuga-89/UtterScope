"""Tests for ASR helpers."""

from __future__ import annotations

from utterscope.asr import resolve_model_path, transcript_from_whisper_result


def test_resolve_model_path_alias() -> None:
    assert (
        resolve_model_path("large-v3-turbo") == "mlx-community/whisper-large-v3-turbo"
    )


def test_resolve_model_path_passthrough() -> None:
    repo = "mlx-community/whisper-tiny"
    assert resolve_model_path(repo) == repo


def test_transcript_from_whisper_result() -> None:
    transcript = transcript_from_whisper_result(
        {
            "text": " Hello world. ",
            "language": "en",
            "segments": [
                {"start": 0.0, "end": 1.0, "text": " Hello"},
                {"start": 1.0, "end": 2.0, "text": " world."},
            ],
        }
    )

    assert transcript.language == "en"
    assert transcript.full_text == "Hello world."
    assert len(transcript.segments) == 2
    assert transcript.segments[0].text == "Hello"
