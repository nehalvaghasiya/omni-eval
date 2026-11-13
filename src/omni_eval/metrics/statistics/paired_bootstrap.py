"""Nonparametric paired bootstrap over common sample IDs, not independent runs."""

from collections.abc import Callable, Sequence

import numpy as np


def paired_bootstrap(
    left: Sequence[float],
    right: Sequence[float],
    *,
    confidence: float = 0.95,
    iterations: int = 1000,
    seed: int = 42,
) -> tuple[float, float] | None:
    if len(left) != len(right):
        raise ValueError("paired bootstrap requires equal observation counts")

    if len(left) < 2:
        return None

    if not 0 < confidence < 1 or iterations < 100:
        raise ValueError("confidence must be (0,1) and iterations >= 100")

    differences = np.asarray(right, dtype=float) - np.asarray(left, dtype=float)
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(left), size=(iterations, len(left)))
    means = differences[indices].mean(axis=1)

    return (
        float(np.quantile(means, (1 - confidence) / 2)),
        float(np.quantile(means, 1 - (1 - confidence) / 2)),
    )


def paired_bootstrap_statistic(
    n: int,
    delta_at_indices: Callable[[list[int]], float],
    *,
    confidence: float = 0.95,
    iterations: int = 1000,
    seed: int = 42,
) -> tuple[float, float] | None:
    if n < 2:
        return None

    if not 0 < confidence < 1 or iterations < 100:
        raise ValueError("invalid confidence or bootstrap iterations")

    rng = np.random.default_rng(seed)
    deltas = [
        delta_at_indices([int(i) for i in rng.integers(0, n, n)])
        for _ in range(iterations)
    ]

    return float(np.quantile(deltas, (1 - confidence) / 2)), float(
        np.quantile(deltas, 1 - (1 - confidence) / 2)
    )
