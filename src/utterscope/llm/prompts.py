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
    return f"""あなたは英語学習者向けのコーチです。
フィードバックはすべて日本語で書いてください。

下の会話全体を分析してください。学習者（student）は {learner_speaker} です。
他の話者は文脈用です。

タスク:
1. 学習者の全体的なパフォーマンスを、日本語で簡潔に要約する（2〜4文）。
2. 学習者の発話に繰り返し見られる傾向（grammar / naturalness / vocabulary）を
   列挙する。複数回現れるもの、または習慣的なものだけを含める。

ルール:
- recurring_patterns は学習者のみを対象にする（相手話者は対象外）。
- summary / label / message は必ず日本語。
- examples は学習者の発話からの英語原文引用。
- 提供された JSON スキーマだけで応答する。

会話:
{transcript}
"""


def build_pass_b_prompt(
    *,
    target: DialogueTurn,
    window: list[DialogueTurn],
    learner_speaker: str,
) -> str:
    context = _format_turns(window)
    return f"""あなたは英語学習者向けのコーチです。
フィードバックはすべて日本語で書いてください。

学習者（student）は {learner_speaker} です。
会話の文脈つきで、学習者の1ターンだけをレビューします。

対象ターン index: {target.index}
対象テキスト: {target.text!r}
対象時間: {target.start:.2f}s – {target.end:.2f}s

周辺コンテキスト（理解用）:
{context}

タスク:
- 対象ターンのみについて、grammar / naturalness / vocabulary の問題を見つける。
- ターン全体なら scope="turn"、短い箇所なら scope="phrase"。
- excerpt は対象テキストからの英語原文をそのままコピー（書き換え禁止）。
- message は日本語。suggestion は必要ならより自然な英語の言い換え。
- 問題がなければ issues は空リスト。

ルール:
- 他の話者を批判しない。
- excerpt に対象テキストにない語を作らない。
- 提供された JSON スキーマだけで応答する。
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
