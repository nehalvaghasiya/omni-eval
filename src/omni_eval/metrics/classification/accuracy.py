from collections.abc import Sequence


def accuracy(labels: Sequence[str], predictions: Sequence[str]) -> float:
    if not labels or len(labels) != len(predictions):
        raise ValueError("expected equal nonempty label vectors")

    return sum(a == b for a, b in zip(labels, predictions, strict=True)) / len(labels)
