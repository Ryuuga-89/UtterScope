"""Tests for dialogue turns and LLM feedback orchestration."""

from __future__ import annotations

from utterscope.llm import generate_feedback_document
from utterscope.llm.align import issues_for_turn
from utterscope.llm.schemas import PassAPattern, PassAResult, PassBIssue, PassBResult
from utterscope.llm.turns import (
    build_dialogue_turns,
    context_window,
    learner_turn_indices,
)
from utterscope.models import Segment, Transcript


def test_build_dialogue_turns_merges_adjacent_same_speaker() -> None:
    transcript = Transcript(
        language="en",
        segments=[
            Segment(start=0.0, end=1.0, text="Hello", speaker="SPEAKER_00"),
            Segment(start=1.02, end=2.0, text="there", speaker="SPEAKER_00"),
            Segment(start=2.5, end=3.5, text="Hi", speaker="SPEAKER_01"),
        ],
        full_text="Hello there Hi",
    )
    turns = build_dialogue_turns(transcript, learner_speaker="SPEAKER_01")
    assert len(turns) == 2
    assert turns[0].text == "Hello there"
    assert turns[0].role == "other"
    assert turns[1].role == "student"
    assert turns[1].index == 1


def test_context_window_and_learner_indices() -> None:
    transcript = Transcript(
        language="en",
        segments=[
            Segment(start=0.0, end=1.0, text="A", speaker="SPEAKER_00"),
            Segment(start=1.0, end=2.0, text="B", speaker="SPEAKER_01"),
            Segment(start=2.0, end=3.0, text="C", speaker="SPEAKER_00"),
            Segment(start=3.0, end=4.0, text="D", speaker="SPEAKER_01"),
        ],
        full_text="A B C D",
    )
    turns = build_dialogue_turns(transcript, learner_speaker="SPEAKER_01")
    assert learner_turn_indices(turns, learner_speaker="SPEAKER_01") == [1, 3]
    window = context_window(turns, center_index=1, radius=1)
    assert [turn.index for turn in window] == [0, 1, 2]


def test_issues_for_turn_drops_unknown_excerpts() -> None:
    from utterscope.llm.turns import DialogueTurn

    turn = DialogueTurn(
        index=0,
        speaker_id="SPEAKER_01",
        start=1.0,
        end=2.0,
        text="I go to school yesterday",
        role="student",
    )
    drafts = [
        PassBIssue(
            category="grammar",
            excerpt="I go to school yesterday",
            message="Use past tense.",
            suggestion="I went to school yesterday",
        ),
        PassBIssue(
            category="vocabulary",
            excerpt="not in the turn",
            message="Should be dropped",
        ),
    ]
    issues = issues_for_turn(drafts, turn=turn)
    assert len(issues) == 1
    assert issues[0].turn_index == 0
    assert issues[0].start == 1.0
    assert issues[0].suggestion == "I went to school yesterday"


class FakeFeedbackBackend:
    def run_pass_a(self, turns, *, learner_speaker: str, model: str):
        assert learner_speaker == "SPEAKER_01"
        assert model == "gemini-3.8-flash"
        assert turns
        return PassAResult(
            summary="The learner communicated clearly overall.",
            recurring_patterns=[
                PassAPattern(
                    category="grammar",
                    label="tense mix-ups",
                    count=2,
                    examples=["I go yesterday"],
                    message="Past events often use present tense.",
                )
            ],
        )

    def run_pass_b(
        self,
        *,
        target,
        window,
        learner_speaker: str,
        model: str,
    ):
        assert learner_speaker == "SPEAKER_01"
        assert target.speaker_id == "SPEAKER_01"
        assert any(turn.index == target.index for turn in window)
        return PassBResult(
            issues=[
                PassBIssue(
                    category="naturalness",
                    scope="turn",
                    excerpt=target.text,
                    message="Sounds a bit stiff.",
                    suggestion=None,
                )
            ]
        )


def test_generate_feedback_document_with_fake_backend() -> None:
    transcript = Transcript(
        language="en",
        segments=[
            Segment(start=0.0, end=1.0, text="Hello", speaker="SPEAKER_00"),
            Segment(start=1.0, end=2.0, text="Hi there", speaker="SPEAKER_01"),
            Segment(start=2.0, end=3.0, text="How are you", speaker="SPEAKER_00"),
            Segment(start=3.0, end=4.0, text="I am fine", speaker="SPEAKER_01"),
        ],
        full_text="Hello Hi there How are you I am fine",
    )
    labels: list[str] = []
    document = generate_feedback_document(
        transcript,
        learner_speaker="SPEAKER_01",
        source_audio="lesson.mp3",
        asr_model="tiny",
        backend=FakeFeedbackBackend(),
        on_progress=labels.append,
    )
    assert document.summary.startswith("The learner")
    assert len(document.recurring_patterns) == 1
    assert len(document.issues) == 2
    assert document.issues[0].excerpt == "Hi there"
    assert document.issues[1].excerpt == "I am fine"
    assert labels[0] == "pass A: lesson overview"
    assert "pass B: learner turn 1/2" in labels
    assert "pass B: learner turn 2/2" in labels
