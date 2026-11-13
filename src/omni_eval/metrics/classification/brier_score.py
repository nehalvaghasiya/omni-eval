def brier_score(
    labels: list[str],
    class_order: list[str],
    probabilities: list[list[float]],
    *,
    positive_class: str | None = None,
) -> float:
    """Binary: positive probability squared error; multiclass: unnormalized sum across classes."""
    if not labels:
        raise ValueError("empty labels")

    if len(class_order) == 2 and positive_class is not None:
        if positive_class not in class_order:
            raise ValueError("positive_class not in probability labels")

        index = class_order.index(positive_class)

        return sum(
            (row[index] - float(label == positive_class)) ** 2
            for label, row in zip(labels, probabilities, strict=True)
        ) / len(labels)

    return sum(
        sum(
            (p - float(label == cls)) ** 2
            for cls, p in zip(class_order, row, strict=True)
        )
        for label, row in zip(labels, probabilities, strict=True)
    ) / len(labels)
