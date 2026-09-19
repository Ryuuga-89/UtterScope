"""Write analysis artifacts to disk."""

from __future__ import annotations

from pathlib import Path

from utterscope.models import AnalysisDocument, TranscriptDocument


def write_transcript_document(
    document: TranscriptDocument, path: Path
) -> Path:
    """Serialize ``document`` as JSON and return ``path``."""
    return _write_json(document, path)


def write_analysis_document(
    document: AnalysisDocument, path: Path
) -> Path:
    """Serialize ``document`` as JSON and return ``path``."""
    return _write_json(document, path)


def _write_json(
    document: TranscriptDocument | AnalysisDocument, path: Path
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        document.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    return path
