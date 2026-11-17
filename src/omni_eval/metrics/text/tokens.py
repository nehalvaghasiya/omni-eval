"""Multiset token overlap for extraction-like QA."""

from collections import Counter

from omni_eval.metrics.text.normalize import normalize


def rates(actual: str, expected: str) -> tuple[float, float, float]:
    a = normalize(actual).split()
    b = normalize(expected).split()
    if not a and (not b):
        return (1.0, 1.0, 1.0)
    intersection = sum((Counter(a) & Counter(b)).values())
    precision = intersection / len(a) if a else 0.0
    recall = intersection / len(b) if b else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return (precision, recall, f1)
