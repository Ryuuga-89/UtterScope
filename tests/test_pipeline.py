"""Tests for the analyze pipeline stub."""

from __future__ import annotations

from pathlib import Path

from utterscope.models import AnalyzeRequest
from utterscope.pipeline import TRANSCRIPT_FILENAME, run


def test_run_creates_output_dir_and_empty_document(tmp_path: Path) -> None:
    audio = tmp_path / "lesson.mp3"
    audio.write_bytes(b"fake")
    output_dir = tmp_path / "results"

    result = run(
        AnalyzeRequest(
            audio_path=audio,
            model="large-v3-turbo",
            output_dir=output_dir,
            llm=True,
        )
    )

    assert output_dir.is_dir()
    assert result.transcript_path == output_dir / TRANSCRIPT_FILENAME
    assert not result.transcript_path.exists()
    assert result.document.source_audio == "lesson.mp3"
    assert result.document.model == "large-v3-turbo"
    assert result.document.transcript.segments == []
    assert result.document.transcript.full_text == ""
    assert result.duration_seconds is None
