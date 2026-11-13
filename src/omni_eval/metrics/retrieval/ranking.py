"""Stable rank semantics: explicit ranks, ascending; ties are invalid upstream."""

import math
from typing import Any


def ranking_metric(
    name: str, ranked: list[str], relevant: set[str], k: int = 10
) -> float:
    if not relevant:
        raise ValueError("relevance judgments require >=1 relevant ID")

    if k < 1:
        raise ValueError("k must be >= 1")

    top = ranked[:k]
    hit_count = sum(chunk in relevant for chunk in top)

    if name == "precision_at_k":
        return hit_count / k  # absent results count as unretrieved at fixed K

    if name in ("recall_at_k", "evidence_coverage"):
        return hit_count / len(relevant)

    if name in ("hit_at_k", "success_at_k"):
        return float(hit_count > 0)

    if name == "mrr":
        return next(
            (1 / rank for rank, chunk in enumerate(ranked, 1) if chunk in relevant), 0.0
        )

    if name in ("map", "average_precision"):
        hits = 0
        accumulated = 0.0

        for rank, chunk in enumerate(ranked, 1):
            if chunk in relevant:
                hits += 1
                accumulated += hits / rank

        return accumulated / len(relevant)

    if name == "ndcg_at_k":
        dcg = sum(
            1 / math.log2(rank + 1)
            for rank, chunk in enumerate(top, 1)
            if chunk in relevant
        )
        ideal = sum(
            1 / math.log2(rank + 1) for rank in range(1, min(k, len(relevant)) + 1)
        )

        return dcg / ideal

    if name == "r_precision":
        return sum(chunk in relevant for chunk in ranked[: len(relevant)]) / len(
            relevant
        )

    raise ValueError(f"unknown ranking metric {name}")
