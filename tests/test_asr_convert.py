"""Tests for ASR convert helpers."""

from __future__ import annotations

from utterscope.asr.convert import transcript_from_whisper_result


def test_transcript_from_whisper_result_keeps_speakers() -> None:
    result = {
        "language": "en",
        "text": "Hello Hi",
        "segments": [
            {
                "start": 0.0,
                "end": 1.0,
                "text": " Hello",
                "speaker": "SPEAKER_00",
                "words": [
                    {"word": "Hello", "start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}
                ],
            },
            {
                "start": 1.0,
                "end": 2.0,
                "text": " Hi",
                "words": [
                    {"word": "Hi", "start": 1.0, "end": 2.0, "speaker": "SPEAKER_01"}
                ],
            },
        ],
    }

    transcript = transcript_from_whisper_result(result)

    assert transcript.language == "en"
    assert transcript.segments[0].speaker == "SPEAKER_00"
    assert transcript.segments[1].speaker == "SPEAKER_01"
    assert transcript.full_text == "Hello Hi"


def test_transcript_from_whisper_result_skips_empty_text() -> None:
    result = {
        "segments": [
            {"start": 0.0, "end": 0.5, "text": "  ", "speaker": "SPEAKER_00"},
            {"start": 0.5, "end": 1.0, "text": "Ok", "speaker": "SPEAKER_01"},
        ]
    }
    transcript = transcript_from_whisper_result(result)
    assert len(transcript.segments) == 1
    assert transcript.segments[0].text == "Ok"
