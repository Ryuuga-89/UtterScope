"""pyannote.audio diarization backend."""

from __future__ import annotations

import inspect
import os
import wave
from pathlib import Path

import numpy as np

from utterscope.audio import PreparedAudio
from utterscope.diarization.types import DiarizationResult, SpeakerTurn

# Current pyannote.audio releases use the community pipeline assets.
DEFAULT_PIPELINE = "pyannote/speaker-diarization-community-1"
REQUIRED_MODEL_PAGES = (
    "https://huggingface.co/pyannote/speaker-diarization-community-1",
)


class DiarizationError(Exception):
    """Raised when speaker diarization fails."""


class PyannoteDiarizationBackend:
    """Diarization backend backed by ``pyannote.audio``."""

    def __init__(self, *, pipeline_name: str = DEFAULT_PIPELINE) -> None:
        self._pipeline_name = pipeline_name
        self._pipeline = None

    def diarize(
        self,
        audio: PreparedAudio,
        *,
        num_speakers: int | None = 2,
    ) -> DiarizationResult:
        pipeline = self._load_pipeline()
        file_input = {
            "waveform": _read_waveform_channels_first(audio.path),
            "sample_rate": audio.sample_rate,
        }
        try:
            kwargs: dict[str, int] = {}
            if num_speakers is not None:
                kwargs["num_speakers"] = num_speakers
            output = pipeline(file_input, **kwargs)
        except Exception as exc:
            msg = f"pyannote diarization failed: {exc}"
            raise DiarizationError(msg) from exc

        annotation = _as_annotation(output)
        turns: list[SpeakerTurn] = []
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            turns.append(
                SpeakerTurn(
                    start=float(turn.start),
                    end=float(turn.end),
                    speaker_id=str(speaker),
                )
            )
        return DiarizationResult(turns=turns)

    def _load_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline

        token = os.environ.get("HF_TOKEN") or os.environ.get(
            "HUGGINGFACE_HUB_TOKEN"
        )
        if not token:
            pages = " and ".join(REQUIRED_MODEL_PAGES)
            msg = (
                "Hugging Face token is required for pyannote diarization. "
                f"Set HF_TOKEN in .env after accepting model terms at {pages}"
            )
            raise DiarizationError(msg)

        try:
            from pyannote.audio import Pipeline
        except ImportError as exc:
            msg = (
                "pyannote.audio is not installed; "
                "run `uv sync` to enable speaker diarization"
            )
            raise DiarizationError(msg) from exc

        try:
            pipeline = Pipeline.from_pretrained(
                self._pipeline_name,
                **_auth_kwargs(Pipeline.from_pretrained, token),
            )
        except Exception as exc:
            pages = "\n  - ".join(REQUIRED_MODEL_PAGES)
            msg = (
                f"failed to load {self._pipeline_name}: {exc}. "
                "Confirm HF_TOKEN is valid and accept gated model terms:\n"
                f"  - {pages}"
            )
            raise DiarizationError(msg) from exc

        if pipeline is None:
            pages = "\n  - ".join(REQUIRED_MODEL_PAGES)
            msg = (
                f"failed to load {self._pipeline_name}: "
                "Pipeline.from_pretrained returned None. "
                "Accept gated model terms:\n"
                f"  - {pages}"
            )
            raise DiarizationError(msg)

        self._pipeline = pipeline
        return pipeline


def _auth_kwargs(from_pretrained, token: str) -> dict[str, str]:
    parameters = inspect.signature(from_pretrained).parameters
    if "token" in parameters:
        return {"token": token}
    if "use_auth_token" in parameters:
        return {"use_auth_token": token}
    msg = "Pipeline.from_pretrained does not accept a Hugging Face token argument"
    raise DiarizationError(msg)


def _as_annotation(output: object):
    if hasattr(output, "itertracks"):
        return output
    speaker_diarization = getattr(output, "speaker_diarization", None)
    if speaker_diarization is not None and hasattr(
        speaker_diarization, "itertracks"
    ):
        return speaker_diarization
    msg = "pyannote returned an unexpected diarization output type"
    raise DiarizationError(msg)


def _read_waveform_channels_first(path: Path):
    """Load mono PCM WAV as a (channel, time) float32 torch tensor."""
    import torch

    with wave.open(str(path), "rb") as handle:
        channel_count = handle.getnchannels()
        sample_width = handle.getsampwidth()
        frames = handle.readframes(handle.getnframes())

    if channel_count != 1:
        msg = f"expected mono audio, got {channel_count} channels"
        raise DiarizationError(msg)
    if sample_width != 2:
        msg = f"expected 16-bit PCM audio, got sample width {sample_width}"
        raise DiarizationError(msg)

    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    return torch.from_numpy(samples).unsqueeze(0)
