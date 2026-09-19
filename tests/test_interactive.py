"""Tests for interactive analyze prompts."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from utterscope.cli.app import app
from utterscope.cli.interactive import (
    InteractiveLlmChoice,
    prompt_asr_model,
    prompt_audio_path,
    prompt_llm_choice,
    run_interactive,
)
from utterscope.llm.catalog import default_gemini_model_index
from utterscope.models import (
    DEFAULT_LLM_MODEL,
    AnalysisDocument,
    AnalyzeResult,
    Segment,
    SpeakerRole,
    SpeakingMetrics,
    Transcript,
    TranscriptDocument,
)

runner = CliRunner()


def test_prompt_audio_path_requires_absolute(tmp_path: Path) -> None:
    audio = tmp_path / "lesson.mp3"
    audio.write_bytes(b"x")
    answers = iter(["relative.mp3", str(audio)])
    with patch(
        "utterscope.cli.interactive.Prompt.ask",
        side_effect=lambda *a, **k: next(answers),
    ):
        path = prompt_audio_path()
    assert path == audio.resolve()
    assert path.is_absolute()


def test_prompt_asr_model_uses_radio() -> None:
    tiny = MagicMock()
    tiny.name = "tiny"
    tiny.downloaded = False
    tiny_en = MagicMock()
    tiny_en.name = "tiny.en"
    tiny_en.downloaded = True
    turbo = MagicMock()
    turbo.name = "large-v3-turbo"
    turbo.downloaded = False

    with (
        patch(
            "utterscope.cli.interactive.list_asr_models",
            return_value=[tiny, tiny_en, turbo],
        ),
        patch(
            "utterscope.cli.interactive.ASR_MODEL_CHOICES",
            ("tiny", "tiny.en", "large-v3-turbo"),
        ),
        patch(
            "utterscope.cli.interactive.DEFAULT_ASR_MODEL",
            "tiny.en",
        ),
        patch("utterscope.cli.interactive.select_radio") as select_mock,
    ):
        select_mock.return_value = 2
        assert prompt_asr_model() == "large-v3-turbo"
        assert select_mock.call_args.kwargs["default_index"] == 1


def test_prompt_asr_model_marks_only_downloaded() -> None:
    tiny = MagicMock()
    tiny.name = "tiny"
    tiny.downloaded = True
    heavy = MagicMock()
    heavy.name = "large-v3"
    heavy.downloaded = False

    captured: dict[str, object] = {}

    def fake_select_radio(**kwargs):
        captured["options"] = kwargs["options"]
        return 0

    with (
        patch(
            "utterscope.cli.interactive.list_asr_models",
            return_value=[tiny, heavy],
        ),
        patch(
            "utterscope.cli.interactive.ASR_MODEL_CHOICES",
            ("tiny", "large-v3"),
        ),
        patch(
            "utterscope.cli.interactive.select_radio",
            side_effect=fake_select_radio,
        ),
    ):
        assert prompt_asr_model() == "tiny"

    options = captured["options"]
    assert isinstance(options, list)
    assert options[0].detail == "downloaded"
    assert options[0].detail_ansi != ""
    assert options[1].detail == ""
    assert options[0].label_ansi != options[1].label_ansi


def test_prompt_llm_choice_can_skip() -> None:
    with patch("utterscope.cli.interactive.select_radio", return_value=0):
        choice = prompt_llm_choice()
    assert choice == InteractiveLlmChoice(enabled=False)


def test_prompt_llm_choice_selects_gemini_model() -> None:
    # provider menu: 0=skip, 1=gemini → pick 1; then model index 5
    choices = iter([1, 5])
    with patch(
        "utterscope.cli.interactive.select_radio",
        side_effect=lambda **_: next(choices),
    ):
        choice = prompt_llm_choice()
    assert choice.enabled is True
    assert choice.provider == "gemini"
    assert choice.model == "gemini-3.8-flash"


def test_default_gemini_model_index_matches_constant() -> None:
    assert default_gemini_model_index() == default_gemini_model_index(
        model_id=DEFAULT_LLM_MODEL
    )


def test_no_args_starts_interactive() -> None:
    fake = AnalyzeResult(
        document=TranscriptDocument(
            source_audio="a.mp3",
            model="tiny",
            transcript=Transcript(
                language="en",
                segments=[Segment(start=0, end=1, text="Hi", speaker="SPEAKER_00")],
                full_text="Hi",
            ),
        ),
        transcript_path=Path("transcript.json"),
        learner_speaker="SPEAKER_00",
        analysis_path=Path("analysis.json"),
        analysis_document=AnalysisDocument(
            source_audio="a.mp3",
            model="tiny",
            learner_speaker="SPEAKER_00",
            speakers=[SpeakerRole(speaker_id="SPEAKER_00", role="student")],
            metrics=SpeakingMetrics(
                speaking_time_seconds=1.0,
                speaking_ratio=1.0,
                wpm=60.0,
                turn_count=1,
                average_turn_seconds=1.0,
                pause_count=0,
                long_pause_count=0,
                filler_count=0,
            ),
        ),
    )
    with patch(
        "utterscope.cli.app.run_interactive",
        return_value=fake,
    ) as mocked:
        result = runner.invoke(app, [])
    assert result.exit_code == 0
    mocked.assert_called_once()
    assert "Transcript →" in result.output


def test_run_interactive_skips_llm(tmp_path: Path, sample_audio: Path) -> None:
    audio = sample_audio
    fake_result = AnalyzeResult(
        document=TranscriptDocument(
            source_audio=audio.name,
            model="tiny",
            transcript=Transcript(
                language="en",
                segments=[
                    Segment(
                        start=0,
                        end=1,
                        text="Hello",
                        speaker="SPEAKER_01",
                    )
                ],
                full_text="Hello",
            ),
        ),
        transcript_path=tmp_path / "transcript.json",
        learner_speaker="SPEAKER_01",
        analysis_path=tmp_path / "analysis.json",
        analysis_document=AnalysisDocument(
            source_audio=audio.name,
            model="tiny",
            learner_speaker="SPEAKER_01",
            speakers=[
                SpeakerRole(speaker_id="SPEAKER_01", role="student"),
            ],
            metrics=SpeakingMetrics(
                speaking_time_seconds=1.0,
                speaking_ratio=1.0,
                wpm=60.0,
                turn_count=1,
                average_turn_seconds=1.0,
                pause_count=0,
                long_pause_count=0,
                filler_count=0,
            ),
        ),
    )

    with (
        patch(
            "utterscope.cli.interactive.prompt_audio_path",
            return_value=audio,
        ),
        patch(
            "utterscope.cli.interactive.prompt_asr_model",
            return_value="tiny",
        ),
        patch(
            "utterscope.cli.interactive.run_pipeline",
            return_value=fake_result,
        ) as pipeline,
        patch(
            "utterscope.cli.interactive.prompt_llm_choice",
            return_value=InteractiveLlmChoice(enabled=False),
        ),
    ):
        result = run_interactive(output_dir=tmp_path)

    assert result.feedback_path is None
    pipeline.assert_called_once()
    request = pipeline.call_args.args[0]
    assert request.llm is False
    assert request.model == "tiny"
