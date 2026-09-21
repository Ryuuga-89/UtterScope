"""Japanese display labels for human-facing reports."""

from __future__ import annotations

CATEGORY_LABELS: dict[str, str] = {
    "grammar": "文法",
    "naturalness": "自然さ",
    "vocabulary": "語彙",
}

SEVERITY_LABELS: dict[str, str] = {
    "low": "低",
    "medium": "中",
    "high": "高",
}

ROLE_LABELS: dict[str, str] = {
    "student": "学習者",
    "other": "相手",
}


def category_label(category: str) -> str:
    return CATEGORY_LABELS.get(category, category)


def severity_label(severity: str) -> str:
    return SEVERITY_LABELS.get(severity, severity)


def role_label(role: str | None) -> str:
    if role is None:
        return "話者"
    return ROLE_LABELS.get(role, role)


def format_timestamp(seconds: float) -> str:
    """Format seconds as ``m:ss`` or ``h:mm:ss``."""
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"
