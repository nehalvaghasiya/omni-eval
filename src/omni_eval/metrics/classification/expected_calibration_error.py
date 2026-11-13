def calibration_error(
    labels: list[str],
    class_order: list[str],
    probabilities: list[list[float]],
    *,
    bins: int = 10,
    maximum: bool = False,
) -> float:
    """Equal-width top-label confidence bins; right endpoint 1 belongs to final bin."""
    if bins < 1:
        raise ValueError("bins must be positive")

    groups: list[list[tuple[float, float]]] = [[] for _ in range(bins)]

    for label, row in zip(labels, probabilities, strict=True):
        best = max(range(len(row)), key=lambda i: row[i])
        confidence = row[best]
        index = min(int(confidence * bins), bins - 1)
        groups[index].append((confidence, float(class_order[best] == label)))

    gaps = [
        abs(
            sum(x for x, _ in group) / len(group)
            - sum(y for _, y in group) / len(group)
        )
        if group
        else 0.0
        for group in groups
    ]

    if maximum:
        return max(gaps)

    return sum(
        gap * len(group) / len(labels) for gap, group in zip(gaps, groups, strict=True)
    )
