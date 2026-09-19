"""Speaker diarization."""

from utterscope.diarization.assign import assign_speakers
from utterscope.diarization.base import DiarizationBackend
from utterscope.diarization.pyannote_backend import (
    DEFAULT_PIPELINE,
    DiarizationError,
    PyannoteDiarizationBackend,
)
from utterscope.diarization.select import (
    FixedLearnerSelector,
    LearnerSelector,
    SpeakerPreview,
    build_speaker_previews,
)
from utterscope.diarization.types import DiarizationResult, SpeakerTurn

__all__ = [
    "DEFAULT_PIPELINE",
    "DiarizationBackend",
    "DiarizationError",
    "DiarizationResult",
    "FixedLearnerSelector",
    "LearnerSelector",
    "PyannoteDiarizationBackend",
    "SpeakerPreview",
    "SpeakerTurn",
    "assign_speakers",
    "build_speaker_previews",
]
