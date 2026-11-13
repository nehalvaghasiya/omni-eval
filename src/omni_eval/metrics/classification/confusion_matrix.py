from typing import Any

import sklearn.metrics as sk


def confusion_details(labels: list[str], predictions: list[str]) -> dict[str, Any]:
    classes = sorted(set(labels) | set(predictions))
    matrix = sk.confusion_matrix(labels, predictions, labels=classes).tolist()

    return {"labels": classes, "matrix": matrix}
