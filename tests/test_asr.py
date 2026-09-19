"""Tests for ASR model catalog and download detection."""

from __future__ import annotations

from pathlib import Path

from utterscope.asr.models import (
    ASR_MODEL_CHOICES,
    DEFAULT_ASR_MODEL,
    is_model_downloaded,
    list_asr_models,
    resolve_model_path,
)


def test_resolve_model_path_alias() -> None:
    assert (
        resolve_model_path("large-v3-turbo") == "mlx-community/whisper-large-v3-turbo"
    )


def test_resolve_model_path_passthrough() -> None:
    repo = "org/custom-whisper"
    assert resolve_model_path(repo) == repo


def test_default_asr_model_is_in_choices() -> None:
    assert DEFAULT_ASR_MODEL in ASR_MODEL_CHOICES


def test_is_model_downloaded_detects_snapshot(tmp_path: Path) -> None:
    repo = resolve_model_path("tiny")
    snapshot = tmp_path / f"models--{repo.replace('/', '--')}" / "snapshots" / "abc"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")

    assert is_model_downloaded("tiny", cache_dir=tmp_path) is True
    assert is_model_downloaded("base", cache_dir=tmp_path) is False


def test_list_asr_models_includes_status(tmp_path: Path) -> None:
    models = list_asr_models(cache_dir=tmp_path)
    assert [model.name for model in models] == list(ASR_MODEL_CHOICES)
    assert all(model.downloaded is False for model in models)
    assert models[-1].status_label == "not downloaded"
