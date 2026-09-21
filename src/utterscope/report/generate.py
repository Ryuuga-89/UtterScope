"""Write Markdown and HTML reports for a completed analyze run."""

from __future__ import annotations

from pathlib import Path

from utterscope.models import (
    AnalysisDocument,
    FeedbackDocument,
    TranscriptDocument,
)
from utterscope.report.html import render_html
from utterscope.report.markdown import render_markdown
from utterscope.report.view import build_report_view

REPORT_MD_FILENAME = "report.md"
REPORT_HTML_FILENAME = "report.html"


def write_reports(
    output_dir: Path,
    *,
    transcript_document: TranscriptDocument,
    analysis_document: AnalysisDocument,
    feedback_document: FeedbackDocument | None = None,
    duration_seconds: float | None = None,
    audio_filename: str | None = None,
) -> tuple[Path, Path]:
    """Render and write ``report.md`` and ``report.html`` under ``output_dir``.

    Returns ``(report_md_path, report_html_path)``.
    """
    view = build_report_view(
        transcript_document,
        analysis_document,
        feedback_document,
        duration_seconds=duration_seconds,
        audio_filename=audio_filename,
    )
    md_path = output_dir / REPORT_MD_FILENAME
    html_path = output_dir / REPORT_HTML_FILENAME
    md_path.write_text(render_markdown(view), encoding="utf-8")
    html_path.write_text(render_html(view), encoding="utf-8")
    return md_path, html_path
