"""Tests for the analyze pipeline."""

from __future__ import annotations

from pathlib import Path

from utterscope.audio import PREPARED_FILENAME, PreparedAudio
from utterscope.models import AnalyzeRequest, Segment, Transcript
from utterscope.pipeline import TRANSCRIPT_FILENAME, run
from utterscope.pipeline.analyze import WORK_DIRNAME


class FakeAsrBackend:
    def transcribe(self, audio: PreparedAudio, *, model: str) -> Transcript:
        assert audio.path.is_file()
        return Transcript(
            language="en",
            segments=[Segment(start=0.0, end=1.0, text="Hello")],
            full_text="Hello",
        )


class RecordingProgress:
    def __init__(self) -> None:
        self.events: list[str] = []

    def on_prepare_audio(self, duration_seconds: float) -> None:
        assert duration_seconds > 0
        self.events.append("prepare_audio")

    def on_transcribe(self, segment_count: int) -> None:
        assert segment_count == 1
        self.events.append("transcribe")

    def on_write_transcript(self, path: Path) -> None:
        assert path.is_file()
        self.events.append("write_transcript")


def test_run_transcribes_and_writes_json(
    sample_audio: Path, tmp_path: Path
) -> None:
    output_dir = tmp_path / "results"
    progress = RecordingProgress()

    result = run(
        AnalyzeRequest(
            audio_path=sample_audio,
            model="tiny",
            output_dir=output_dir,
            llm=False,
        ),
        asr=FakeAsrBackend(),
        progress=progress,
    )

    prepared = output_dir / WORK_DIRNAME / PREPARED_FILENAME
    assert prepared.is_file()
    assert result.transcript_path == output_dir / TRANSCRIPT_FILENAME
    assert result.transcript_path.is_file()
    assert result.document.transcript.full_text == "Hello"
    assert result.duration_seconds is not None
    assert result.duration_seconds > 0
    assert progress.events == [
        "prepare_audio",
        "transcribe",
        "write_transcript",
    ]
