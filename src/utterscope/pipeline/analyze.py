"""Analyze pipeline orchestration."""

from __future__ import annotations

from utterscope.asr import AsrBackend, MlxWhisperBackend
from utterscope.audio import prepare_audio
from utterscope.models import AnalyzeRequest, AnalyzeResult, TranscriptDocument
from utterscope.report import write_transcript_document

TRANSCRIPT_FILENAME = "transcript.json"
WORK_DIRNAME = ".utterscope"


def run(
    request: AnalyzeRequest,
    *,
    asr: AsrBackend | None = None,
) -> AnalyzeResult:
    """Run the analyze pipeline for a single audio file.

    Prepares audio, runs ASR, and writes ``transcript.json``.
    ``request.llm`` is ignored in v0.1.
    """
    request.output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = request.output_dir / WORK_DIRNAME
    prepared = prepare_audio(request.audio_path, work_dir)

    backend = asr or MlxWhisperBackend()
    transcript = backend.transcribe(prepared, model=request.model)

    document = TranscriptDocument(
        source_audio=request.audio_path.name,
        model=request.model,
        transcript=transcript,
    )
    transcript_path = write_transcript_document(
        document,
        request.output_dir / TRANSCRIPT_FILENAME,
    )

    return AnalyzeResult(
        document=document,
        transcript_path=transcript_path,
        duration_seconds=prepared.duration_seconds,
    )
