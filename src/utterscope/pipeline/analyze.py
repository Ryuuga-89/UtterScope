"""Analyze pipeline orchestration."""

from __future__ import annotations

from utterscope.models import (
    AnalyzeRequest,
    AnalyzeResult,
    Transcript,
    TranscriptDocument,
)

TRANSCRIPT_FILENAME = "transcript.json"


def run(request: AnalyzeRequest) -> AnalyzeResult:
    """Run the analyze pipeline for a single audio file.

    v0.1 wiring stub: prepares the output directory and returns an empty
    transcript document. Audio preprocessing and ASR are not implemented yet.
    ``request.llm`` is accepted for CLI compatibility and ignored.
    """
    request.output_dir.mkdir(parents=True, exist_ok=True)

    document = TranscriptDocument(
        source_audio=request.audio_path.name,
        model=request.model,
        transcript=Transcript(),
    )
    transcript_path = request.output_dir / TRANSCRIPT_FILENAME

    return AnalyzeResult(
        document=document,
        transcript_path=transcript_path,
        duration_seconds=None,
    )
