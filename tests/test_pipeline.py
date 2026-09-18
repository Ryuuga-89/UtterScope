"""Tests for the analyze pipeline."""

from __future__ import annotations

from pathlib import Path

from utterscope.audio import PREPARED_FILENAME
from utterscope.models import AnalyzeRequest
from utterscope.pipeline import TRANSCRIPT_FILENAME, run
from utterscope.pipeline.analyze import WORK_DIRNAME


def test_run_prepares_audio_and_returns_empty_document(
    sample_audio: Path, tmp_path: Path
) -> None:
    output_dir = tmp_path / "results"

    result = run(
        AnalyzeRequest(
            audio_path=sample_audio,
            model="large-v3-turbo",
            output_dir=output_dir,
            llm=True,
        )
    )

    prepared = output_dir / WORK_DIRNAME / PREPARED_FILENAME
    assert prepared.is_file()
    assert result.transcript_path == output_dir / TRANSCRIPT_FILENAME
    assert not result.transcript_path.exists()
    assert result.document.source_audio == sample_audio.name
    assert result.duration_seconds is not None
    assert result.duration_seconds > 0
