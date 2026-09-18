"""Analyze pipeline orchestration."""

from __future__ import annotations

from utterscope.audio import prepare_audio
from utterscope.models import (
    AnalyzeRequest,
    AnalyzeResult,
    Transcript,
    TranscriptDocument,
)

TRANSCRIPT_FILENAME = "transcript.json"
WORK_DIRNAME = ".utterscope"


def run(request: AnalyzeRequest) -> AnalyzeResult:
    """Run the analyze pipeline for a single audio file.

    Prepares audio via FFmpeg, then returns an empty transcript document.
    ASR is not implemented yet. ``request.llm`` is ignored in v0.1.
    """
    request.output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = request.output_dir / WORK_DIRNAME
    prepared = prepare_audio(request.audio_path, work_dir)

    document = TranscriptDocument(
        source_audio=request.audio_path.name,
        model=request.model,
        transcript=Transcript(),
    )
    transcript_path = request.output_dir / TRANSCRIPT_FILENAME

    return AnalyzeResult(
        document=document,
        transcript_path=transcript_path,
        duration_seconds=prepared.duration_seconds,
    )
