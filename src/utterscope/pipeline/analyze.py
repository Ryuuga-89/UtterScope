"""Analyze pipeline orchestration."""

from __future__ import annotations

from typing import Protocol

from utterscope.asr import AsrBackend, MlxWhisperBackend
from utterscope.audio import prepare_audio
from utterscope.diarization import (
    DiarizationBackend,
    FixedLearnerSelector,
    LearnerSelector,
    PyannoteDiarizationBackend,
    assign_speakers,
    build_speaker_previews,
)
from utterscope.llm import FeedbackBackend, generate_feedback_document
from utterscope.metrics import build_speaker_roles, compute_speaking_metrics
from utterscope.models import (
    AnalysisDocument,
    AnalyzeRequest,
    AnalyzeResult,
    FeedbackDocument,
    TranscriptDocument,
)
from utterscope.report import (
    write_analysis_document,
    write_feedback_document,
    write_reports,
    write_transcript_document,
)
from utterscope.runtime import silence_third_party
from utterscope.vad import SileroVadBackend, VadBackend

TRANSCRIPT_FILENAME = "transcript.json"
ANALYSIS_FILENAME = "analysis.json"
FEEDBACK_FILENAME = "feedback.json"
WORK_DIRNAME = ".utterscope"
DEFAULT_NUM_SPEAKERS = 2


class PipelineProgress(Protocol):
    """Sink for in-progress and completed pipeline steps."""

    def begin(self, step: str) -> None:
        """Announce that ``step`` has started and the user should wait."""

    def end(self, step: str, detail: str = "") -> None:
        """Announce that ``step`` finished."""

    def update(self, step: str) -> None:
        """Update the waiting label for the current step."""


def run(
    request: AnalyzeRequest,
    *,
    asr: AsrBackend | None = None,
    vad: VadBackend | None = None,
    diarization: DiarizationBackend | None = None,
    learner_selector: LearnerSelector | None = None,
    feedback_backend: FeedbackBackend | None = None,
    progress: PipelineProgress | None = None,
    quiet: bool = True,
) -> AnalyzeResult:
    """Run the analyze pipeline for a single audio file.

    Prepares audio, detects speech, transcribes, diarizes speakers,
    selects the learner, computes speaking metrics, optionally runs LLM
    feedback, and writes JSON outputs plus Markdown/HTML reports.
    """
    request.output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = request.output_dir / WORK_DIRNAME

    if progress is not None:
        progress.begin("prepare audio")
    prepared = prepare_audio(request.audio_path, work_dir)
    if progress is not None:
        progress.end("prepare audio", f"{prepared.duration_seconds:.1f}s")

    if progress is not None:
        progress.begin("detect speech")
    with silence_third_party(enabled=quiet):
        interval_count = len((vad or SileroVadBackend()).detect(prepared).intervals)
    if progress is not None:
        progress.end("detect speech", f"{interval_count} intervals")

    if progress is not None:
        progress.begin("transcribe")
    with silence_third_party(enabled=quiet):
        transcript = (asr or MlxWhisperBackend()).transcribe(
            prepared,
            model=request.model,
        )
    if progress is not None:
        progress.end("transcribe", f"{len(transcript.segments)} segments")

    if progress is not None:
        progress.begin("identify speakers")
    with silence_third_party(enabled=quiet):
        diarization_result = (diarization or PyannoteDiarizationBackend()).diarize(
            prepared,
            num_speakers=DEFAULT_NUM_SPEAKERS,
        )
        transcript = assign_speakers(transcript, diarization_result)
    if progress is not None:
        progress.end(
            "identify speakers",
            f"{len(diarization_result.speaker_ids())} speakers",
        )

    previews = build_speaker_previews(transcript)
    selector = learner_selector
    if selector is None:
        if request.learner_speaker is not None:
            selector = FixedLearnerSelector(request.learner_speaker)
        else:
            msg = "learner_selector is required when learner_speaker is not set"
            raise ValueError(msg)
    learner_speaker = selector.select_learner(previews)
    request = request.model_copy(update={"learner_speaker": learner_speaker})

    if progress is not None:
        progress.begin("analyze learner speech")
    metrics = compute_speaking_metrics(
        transcript,
        learner_speaker=learner_speaker,
        long_pause_threshold_seconds=request.long_pause_threshold_seconds,
    )
    analysis_document = AnalysisDocument(
        source_audio=request.audio_path.name,
        model=request.model,
        learner_speaker=learner_speaker,
        speakers=build_speaker_roles(
            transcript,
            learner_speaker=learner_speaker,
        ),
        metrics=metrics,
        long_pause_threshold_seconds=request.long_pause_threshold_seconds,
    )
    if progress is not None:
        progress.end(
            "analyze learner speech",
            f"{metrics.wpm:.0f} wpm",
        )

    feedback_document: FeedbackDocument | None = None
    if request.llm:
        if progress is not None:
            progress.begin("generate feedback")

        def on_progress(label: str) -> None:
            if progress is not None:
                progress.update(label)

        feedback_document = generate_feedback_document(
            transcript,
            learner_speaker=learner_speaker,
            source_audio=request.audio_path.name,
            asr_model=request.model,
            llm_model=request.llm_model,
            context_turns=request.feedback_context_turns,
            backend=feedback_backend,
            on_progress=on_progress,
        )
        if progress is not None:
            progress.end(
                "generate feedback",
                f"{len(feedback_document.issues)} issues",
            )

    if progress is not None:
        progress.begin("write results")
    document = TranscriptDocument(
        source_audio=request.audio_path.name,
        model=request.model,
        transcript=transcript,
    )
    transcript_path = write_transcript_document(
        document,
        request.output_dir / TRANSCRIPT_FILENAME,
    )
    analysis_path = write_analysis_document(
        analysis_document,
        request.output_dir / ANALYSIS_FILENAME,
    )
    feedback_path = None
    if feedback_document is not None:
        feedback_path = write_feedback_document(
            feedback_document,
            request.output_dir / FEEDBACK_FILENAME,
        )
    if progress is not None:
        progress.end("write results")

    if progress is not None:
        progress.begin("generate reports")
    source_audio_path = (
        request.output_dir / request.audio_path.name
        if (request.output_dir / request.audio_path.name).is_file()
        else None
    )
    report_md_path, report_html_path = write_reports(
        request.output_dir,
        transcript_document=document,
        analysis_document=analysis_document,
        feedback_document=feedback_document,
        duration_seconds=prepared.duration_seconds,
        audio_filename=(
            source_audio_path.name if source_audio_path is not None else None
        ),
    )
    if progress is not None:
        progress.end("generate reports", "report.md, report.html")

    return AnalyzeResult(
        document=document,
        transcript_path=transcript_path,
        duration_seconds=prepared.duration_seconds,
        learner_speaker=learner_speaker,
        analysis_document=analysis_document,
        analysis_path=analysis_path,
        feedback_document=feedback_document,
        feedback_path=feedback_path,
        run_dir=request.output_dir,
        source_audio_path=source_audio_path,
        report_md_path=report_md_path,
        report_html_path=report_html_path,
    )
