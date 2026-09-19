"""Prompt builders for Gemini feedback passes."""

from __future__ import annotations

import json

from utterscope.llm.turns import DialogueTurn


def build_pass_a_prompt(
    turns: list[DialogueTurn],
    *,
    learner_speaker: str,
) -> str:
    transcript = _format_turns(turns)
    return f"""You are an English-speaking coach for language learners.

Analyze the full conversation below. The learner (student) is {learner_speaker}.
Other speakers are context only.

Tasks:
1. Write a concise summary (2-4 sentences) of the learner's overall
   performance.
2. List recurring patterns in the learner's speech
   (grammar, naturalness, or vocabulary).
   Only include patterns that appear more than once or are clearly habitual.

Rules:
- Focus recurring patterns on the learner, not the interlocutor.
- Use short labels and concrete examples quoted from the learner when possible.
- Respond only via the provided JSON schema.

Conversation:
{transcript}
"""


def build_pass_b_prompt(
    *,
    target: DialogueTurn,
    window: list[DialogueTurn],
    learner_speaker: str,
) -> str:
    context = _format_turns(window)
    return f"""You are an English-speaking coach for language learners.

The learner (student) is {learner_speaker}.
You will review ONE learner turn in conversational context.

Target turn index: {target.index}
Target text: {target.text!r}
Target time range: {target.start:.2f}s – {target.end:.2f}s

Surrounding context (for understanding only):
{context}

Tasks:
- Find grammar, naturalness, and vocabulary issues in the TARGET turn only.
- scope="turn" for whole-turn feedback; scope="phrase" for a shorter
  excerpt inside the target.
- excerpt must be copied verbatim from the target text (do not rewrite it).
- suggestion is an improved alternative when helpful.
- If the target turn is fine, return an empty issues list.

Rules:
- Do not criticize other speakers.
- Do not invent words that are not in the target text for excerpt.
- Respond only via the provided JSON schema.
"""


def _format_turns(turns: list[DialogueTurn]) -> str:
    rows = []
    for turn in turns:
        role = turn.role or "unknown"
        rows.append(
            {
                "index": turn.index,
                "speaker": turn.speaker_id,
                "role": role,
                "start": round(turn.start, 2),
                "end": round(turn.end, 2),
                "text": turn.text,
            }
        )
    return json.dumps(rows, ensure_ascii=False, indent=2)
