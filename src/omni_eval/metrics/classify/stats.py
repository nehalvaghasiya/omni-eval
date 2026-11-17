"""Corpus confusion counts and class metrics."""

from omni_eval.metrics.data import label
from omni_eval.types.case import Case


class Counts:
    def __init__(self, cases: list[Case]):
        self.truth = [label(c, truth=True) for c in cases]
        self.pred = [label(c) for c in cases]
        self.labels = sorted(set(self.truth + self.pred), key=str)
        self.matrix: list[list[int]] = [
            [
                sum(int(t == a and p == b) for t, p in zip(self.truth, self.pred, strict=True))
                for b in self.labels
            ]
            for a in self.labels
        ]

    def support(self, i: int) -> int:
        return sum(self.matrix[i])

    def tp(self, i: int) -> int:
        return self.matrix[i][i]

    def fp(self, i: int) -> int:
        return sum(row[i] for row in self.matrix) - self.tp(i)

    def fn(self, i: int) -> int:
        return self.support(i) - self.tp(i)

    def tn(self, i: int) -> int:
        return sum(map(sum, self.matrix)) - self.tp(i) - self.fp(i) - self.fn(i)

    def rate(self, i: int, kind: str) -> float:
        tp = self.tp(i)
        fp = self.fp(i)
        fn = self.fn(i)
        if kind == "precision":
            return tp / (tp + fp) if tp + fp else 0.0
        if kind == "recall":
            return tp / (tp + fn) if tp + fn else 0.0
        if kind == "f1":
            return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        raise ValueError("unknown rate")

    def aggregate(self, kind: str, average: str) -> float:
        if not self.labels:
            raise ValueError("empty labels")
        if average == "micro":
            tp = sum(self.tp(i) for i in range(len(self.labels)))
            fp = sum(self.fp(i) for i in range(len(self.labels)))
            fn = sum(self.fn(i) for i in range(len(self.labels)))
            if kind == "precision":
                return tp / (tp + fp) if tp + fp else 0
            if kind == "recall":
                return tp / (tp + fn) if tp + fn else 0
            return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0
        values = [self.rate(i, kind) for i in range(len(self.labels))]
        weights = [self.support(i) if average == "weighted" else 1 for i in range(len(self.labels))]
        return sum((v * w for v, w in zip(values, weights, strict=True))) / sum(weights)
