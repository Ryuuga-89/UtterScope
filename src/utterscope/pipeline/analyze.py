"""Analyze pipeline orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from utterscope.asr import AsrBackend, MlxWhisperBackend
from utterscope.audio import prepare_audio
from utterscope.models import AnalyzeRequest, AnalyzeResult, TranscriptDocument
from utterscope.report import write_transcript_document
from utterscope.vad import SileroVadBackend, VadBackend

TRANSCRIPT_FILENAME = "transcript.json"
WORK_DIRNAME = ".utterscope"


class PipelineProgress(Protocol):
    """Optional sink for step completion events."""

    def on_prepare_audio(self, duration_seconds: float) -> None:
        """Called after audio preprocessing finishes."""

    def on_detect_speech(self, interval_count: int) -> None:
        """Called after VAD finishes."""

    def on_transcribe(self, segment_count: int) -> None:
        """Called after ASR finishes."""

    def on_write_transcript(self, path: Path) -> None:
        """Called after transcript.json is written."""


def run(
    request: AnalyzeRequest,
    *,
    asr: AsrBackend | None = None,
    vad: VadBackend | None = None,
    progress: PipelineProgress | None = None,
) -> AnalyzeResult:
    """Run the analyze pipeline for a single audio file.

    Prepares audio, detects speech, runs ASR, and writes ``transcript.json``.
    ``request.llm`` is ignored until LLM analysis lands.
    """
    request.output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = request.output_dir / WORK_DIRNAME
    prepared = prepare_audio(request.audio_path, work_dir)
    if progress is not None:
        progress.on_prepare_audio(prepared.duration_seconds)

    vad_backend = vad or SileroVadBackend()
    # Speech intervals will feed metrics / diarization in later v0.2 work.
    # ASR still runs on the full prepared audio for now.
    interval_count = len(vad_backend.detect(prepared).intervals)
    if progress is not None:
        progress.on_detect_speech(interval_count)

    backend = asr or MlxWhisperBackend()
    transcript = backend.transcribe(prepared, model=request.model)
    if progress is not None:
        progress.on_transcribe(len(transcript.segments))

    document = TranscriptDocument(
        source_audio=request.audio_path.name,
        model=request.model,
        transcript=transcript,
    )
    transcript_path = write_transcript_document(
        document,
        request.output_dir / TRANSCRIPT_FILENAME,
    )
    if progress is not None:
        progress.on_write_transcript(transcript_path)

    return AnalyzeResult(
        document=document,
        transcript_path=transcript_path,
        duration_seconds=prepared.duration_seconds,
    )
