"""Hand-computed retrieval formulas with explicit qrels and rank ordering."""

from math import log2

from omni_eval.metrics.data import ranks, relevant
from omni_eval.types.case import Case


class Rank:
    def __init__(self, case: Case):
        truth = relevant(case)
        if truth is None:
            raise ValueError("relevance judgments missing")
        self.truth = truth
        self.items = ranks(case)
        if len(set(self.items)) != len(self.items):
            raise ValueError("duplicate retrieved document IDs")

    def metric(self, name: str, k: int) -> float:
        if k <= 0:
            raise ValueError("k must be positive")
        cut = self.items[:k]
        matched = [i for i, x in enumerate(cut, 1) if x in self.truth]
        if name == "precision":
            return len(matched) / k
        if name == "recall":
            return len(matched) / len(self.truth) if self.truth else 1.0
        if name in ("hit", "success"):
            return float(bool(matched))
        if name == "mrr":
            return 1 / matched[0] if matched else 0.0
        if name == "map":
            return (
                sum((j / i for j, i in enumerate(matched, 1))) / len(self.truth)
                if self.truth
                else 1.0
            )
        if name == "rprec":
            return (
                len([x for x in self.items[: len(self.truth)] if x in self.truth]) / len(self.truth)
                if self.truth
                else 1.0
            )
        if name == "ndcg":
            got = sum(1 / log2(i + 1) for i in matched)
            ideal = sum(1 / log2(i + 1) for i in range(1, min(k, len(self.truth)) + 1))
            return got / ideal if ideal else 1.0
        raise ValueError("unknown rank metric")
