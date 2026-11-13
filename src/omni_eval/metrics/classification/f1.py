import sklearn.metrics as sk


def classification_f1(
    labels: list[str], predictions: list[str], average: str = "macro"
) -> float:
    return float(sk.f1_score(labels, predictions, average=average, zero_division=0))
