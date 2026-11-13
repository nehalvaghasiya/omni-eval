from collections import Counter

from omni_eval.metrics.shared.tokenize import tokenize


def token_scores(prediction: str, reference: str) -> tuple[float, float, float]:
    a = Counter(tokenize(prediction))
    b = Counter(tokenize(reference))
    common = sum((a & b).values())

    if not a and not b:
        return 1.0, 1.0, 1.0

    precision = common / sum(a.values()) if a else 0.0
    recall = common / sum(b.values()) if b else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return precision, recall, f1
