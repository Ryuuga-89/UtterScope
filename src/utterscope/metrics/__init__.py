"""Deterministic speaking metrics (duration, WPM, pauses, fillers)."""

from utterscope.metrics.compute import build_speaker_roles, compute_speaking_metrics
from utterscope.metrics.fillers import FILLER_PHRASES, FILLER_WORDS, count_fillers

__all__ = [
    "FILLER_PHRASES",
    "FILLER_WORDS",
    "build_speaker_roles",
    "compute_speaking_metrics",
    "count_fillers",
]
