"""English filler tokens and phrases for learner speech analysis."""

from __future__ import annotations

import re

# Single-token fillers (matched after lowercasing / stripping punctuation).
FILLER_WORDS = frozenset(
    {
        "uh",
        "um",
        "umm",
        "uhh",
        "er",
        "erm",
        "ah",
        "eh",
        "hmm",
        "hm",
        "mm",
        "mhm",
        "like",
    }
)

# Multi-word fillers counted via substring scan on normalized text.
FILLER_PHRASES = (
    "you know",
    "i mean",
    "kind of",
    "sort of",
)

_TOKEN_RE = re.compile(r"[a-z0-9']+", re.IGNORECASE)


def tokenize_words(text: str) -> list[str]:
    """Split ``text`` into lowercase alphabetic tokens."""
    return [match.group(0).lower() for match in _TOKEN_RE.finditer(text)]


def count_fillers(text: str) -> int:
    """Count filler words and phrases in ``text`` (English-oriented)."""
    tokens = tokenize_words(text)
    count = sum(1 for token in tokens if token in FILLER_WORDS)

    normalized = " ".join(tokens)
    for phrase in FILLER_PHRASES:
        # Non-overlapping counts of the phrase as whole words.
        needle = phrase
        start = 0
        while True:
            index = normalized.find(needle, start)
            if index < 0:
                break
            before_ok = index == 0 or normalized[index - 1] == " "
            end = index + len(needle)
            after_ok = end == len(normalized) or normalized[end] == " "
            if before_ok and after_ok:
                count += 1
                start = end
            else:
                start = index + 1
    return count
