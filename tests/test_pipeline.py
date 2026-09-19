"""Tests for the analyze pipeline."""

from __future__ import annotations

from pathlib import Path

from utterscope.audio import PREPARED_FILENAME, PreparedAudio
from utterscope.diarization import DiarizationResult, SpeakerTurn
from utterscope.models import AnalyzeRequest, Segment, Transcript
from utterscope.pipeline import TRANSCRIPT_FILENAME, run
from utterscope.pipeline.analyze import WORK_DIRNAME
from utterscope.vad import SpeechInterval, VadResult


class FakeAsrBackend:
    def transcribe(self, audio: PreparedAudio, *, model: str) -> Transcript:
        assert audio.path.is_file()
        return Transcript(
            language="en",
            segments=[
                Segment(start=0.0, end=1.0, text="Hello"),
                Segment(start=1.0, end=2.0, text="Hi"),
            ],
            full_text="Hello Hi",
        )


class FakeVadBackend:
    def detect(self, audio: PreparedAudio) -> VadResult:
        assert audio.path.is_file()
        return VadResult(intervals=[SpeechInterval(start=0.1, end=1.9)])


class FakeDiarizationBackend:
    def diarize(
        self,
        audio: PreparedAudio,
        *,
        num_speakers: int | None = 2,
    ) -> DiarizationResult:
        assert audio.path.is_file()
        assert num_speakers == 2
        return DiarizationResult(
            turns=[
                SpeakerTurn(start=0.0, end=1.0, speaker_id="SPEAKER_00"),
                SpeakerTurn(start=1.0, end=2.0, speaker_id="SPEAKER_01"),
            ]
        )


class RecordingProgress:
    def __init__(self) -> None:
        self.events: list[str] = []

    def begin(self, step: str) -> None:
        self.events.append(f"begin:{step}")

    def end(self, step: str, detail: str = "") -> None:
        self.events.append(f"end:{step}:{detail}")


def test_run_assigns_speakers_and_learner(
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
            learner_speaker="SPEAKER_01",
        ),
        asr=FakeAsrBackend(),
        vad=FakeVadBackend(),
        diarization=FakeDiarizationBackend(),
        progress=progress,
    )

    prepared = output_dir / WORK_DIRNAME / PREPARED_FILENAME
    assert prepared.is_file()
    assert result.transcript_path == output_dir / TRANSCRIPT_FILENAME
    assert result.transcript_path.is_file()
    assert result.learner_speaker == "SPEAKER_01"
    assert result.document.transcript.segments[0].speaker == "SPEAKER_00"
    assert result.document.transcript.segments[1].speaker == "SPEAKER_01"
    assert result.analysis_path == output_dir / "analysis.json"
    assert result.analysis_path.is_file()
    assert result.analysis_document is not None
    assert result.analysis_document.learner_speaker == "SPEAKER_01"
    assert result.analysis_document.metrics.turn_count == 1
    assert progress.events == [
        "begin:prepare audio",
        "end:prepare audio:1.0s",
        "begin:detect speech",
        "end:detect speech:1 intervals",
        "begin:transcribe",
        "end:transcribe:2 segments",
        "begin:identify speakers",
        "end:identify speakers:2 speakers",
        "begin:analyze learner speech",
        "end:analyze learner speech:60 wpm",
        "begin:write results",
        "end:write results:",
    ]
