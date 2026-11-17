"""Memory-efficient token or character edit distance."""

from collections.abc import Sequence
from typing import TypeVar

T = TypeVar("T")


def distance(actual: Sequence[T], expected: Sequence[T]) -> int:
    previous = list(range(len(expected) + 1))
    for i, a in enumerate(actual, 1):
        current = [i] + [0] * len(expected)
        for j, b in enumerate(expected, 1):
            current[j] = min(current[j - 1] + 1, previous[j] + 1, previous[j - 1] + (a != b))
        previous = current
    return previous[-1]
