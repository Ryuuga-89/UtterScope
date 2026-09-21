"""Markdown report renderer."""

from __future__ import annotations

from utterscope.report.view import ReportView


def render_markdown(view: ReportView) -> str:
    """Render ``report.md`` content (Japanese, human-facing)."""
    lines: list[str] = ["# 分析レポート", ""]

    lines.extend(["## 概要", ""])
    lines.append(f"- 音声: `{view.source_audio}`")
    lines.append(f"- ASR モデル: `{view.asr_model}`")
    lines.append(f"- 学習者: `{view.learner_speaker}`")
    if view.duration_label is not None:
        lines.append(f"- 長さ: {view.duration_label}")
    if view.has_feedback and view.llm_model is not None:
        provider = view.llm_provider or "llm"
        lines.append(f"- LLM: `{provider}` / `{view.llm_model}`")
    lines.append("")

    metrics = view.metrics
    lines.extend(
        [
            "## 発話メトリクス",
            "",
            f"- 発話時間: {float(metrics['speaking_time_seconds']):.1f} 秒",
            f"- 発話比率: {float(metrics['speaking_ratio']):.0%}",
            f"- 発話速度: {float(metrics['wpm']):.0f} WPM",
            f"- ターン数: {int(metrics['turn_count'])}",
            (f"- 平均ターン長: {float(metrics['average_turn_seconds']):.1f} 秒"),
            f"- ポーズ数: {int(metrics['pause_count'])}",
            (
                f"- 長いポーズ数: {int(metrics['long_pause_count'])}"
                f"（閾値 {view.long_pause_threshold_seconds:.1f} 秒）"
            ),
            f"- フィラー数: {int(metrics['filler_count'])}",
            "",
        ]
    )

    if view.has_feedback:
        lines.extend(["## フィードバック", ""])
        if view.summary:
            lines.extend(["### 全体のまとめ", "", view.summary, ""])
        if view.recurring_patterns:
            lines.extend(["### 繰り返しパターン", ""])
            for pattern in view.recurring_patterns:
                lines.append(
                    f"- **{pattern.category_label}** · "
                    f"{pattern.label}（{pattern.count} 回）"
                )
                lines.append(f"  - {pattern.message}")
                if pattern.examples:
                    examples = " / ".join(f"「{ex}」" for ex in pattern.examples)
                    lines.append(f"  - 例: {examples}")
            lines.append("")
        if view.issues:
            lines.extend(["### 指摘一覧", ""])
            for issue in view.issues:
                lines.append(
                    f"- **[{issue.severity_label}] {issue.category_label}** "
                    f"· ターン {issue.turn_index} "
                    f"({issue.start_label}–{issue.end_label})"
                )
                lines.append(f"  - 引用: 「{issue.excerpt}」")
                lines.append(f"  - {issue.message}")
                if issue.suggestion:
                    lines.append(f"  - 提案: 「{issue.suggestion}」")
            lines.append("")
    else:
        lines.extend(
            [
                "## フィードバック",
                "",
                "LLM フィードバックは実行されていません。",
                "",
            ]
        )

    lines.extend(["## 対話トランスクリプト", ""])
    if not view.turns:
        lines.append("（発話ターンなし）")
        lines.append("")
    else:
        for turn in view.turns:
            who = turn.role_label
            lines.append(
                f"### ターン {turn.index} · {who} · {turn.start_label}–{turn.end_label}"
            )
            lines.append("")
            lines.append(turn.text or "（無音）")
            lines.append("")
            for issue in turn.issues:
                lines.append(
                    f"> **[{issue.severity_label}] "
                    f"{issue.category_label}** — {issue.message}"
                )
                if issue.suggestion:
                    lines.append(f"> 提案: 「{issue.suggestion}」")
                lines.append("")

    return "\n".join(lines).rstrip() + "\n"
