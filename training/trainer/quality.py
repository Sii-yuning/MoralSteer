"""Rollout quality checks (Layer 2 of the 3-layer defense in the proposal).

A student rollout fails the check if it (a) is too short, (b) starts with a
refusal phrase, (c) contains a safety-disclaimer suffix, or (d) is degenerate
(>=k repeats of the same trigram).
"""

from __future__ import annotations

import re
from collections import Counter

REFUSAL_PATTERNS = [
    r"\bi (?:cannot|can'?t|will not|won'?t)\b",
    r"\bi'?m (?:sorry|unable|not able)\b",
    r"\bas an? (?:ai|assistant|language model)\b",
    r"\bi (?:must|need to) (?:decline|refuse)\b",
    r"\bi don'?t feel comfortable\b",
]
DISCLAIMER_PATTERNS = [
    r"this is (?:just )?(?:a )?role[- ]?play",
    r"(?:remember|note|please note|disclaimer)[, :]",
    r"in (?:real|actual) life",
    r"i do not (?:endorse|condone|support)",
]

_REFUSAL_RE = re.compile("|".join(REFUSAL_PATTERNS), re.IGNORECASE)
_DISCLAIMER_RE = re.compile("|".join(DISCLAIMER_PATTERNS), re.IGNORECASE)


def is_refusal(text: str) -> bool:
    head = text[:240]
    return _REFUSAL_RE.search(head) is not None


def has_disclaimer(text: str) -> bool:
    tail = text[-240:]
    return _DISCLAIMER_RE.search(tail) is not None


def trigram_repeat_ratio(text: str) -> float:
    tokens = re.findall(r"\w+", text.lower())
    if len(tokens) < 6:
        return 0.0
    trigrams = [tuple(tokens[i : i + 3]) for i in range(len(tokens) - 2)]
    counts = Counter(trigrams)
    max_count = max(counts.values())
    return max_count / len(trigrams)


def passes_quality(text: str, min_chars: int = 20, max_repeat_ratio: float = 0.35) -> tuple[bool, str]:
    """Return (passes, reason). reason is empty when passes is True."""
    if len(text.strip()) < min_chars:
        return False, "too_short"
    if is_refusal(text):
        return False, "refusal"
    if has_disclaimer(text):
        return False, "disclaimer"
    if trigram_repeat_ratio(text) > max_repeat_ratio:
        return False, "repetition"
    return True, ""
