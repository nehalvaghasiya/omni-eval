from math import erfc, sqrt


def mcnemar_test(
    left_correct: list[bool], right_correct: list[bool]
) -> dict[str, float | int | None]:
    """McNemar continuity-corrected large-sample chi-square; no p-value below 10 discordant pairs."""
    if len(left_correct) != len(right_correct):
        raise ValueError("must be paired")

    b = sum(a and not c for a, c in zip(left_correct, right_correct, strict=True))
    c = sum(not a and d for a, d in zip(left_correct, right_correct, strict=True))
    n = b + c

    if n < 10:
        return {"left_only": b, "right_only": c, "p_value": None}

    statistic = (abs(b - c) - 1) ** 2 / n

    return {"left_only": b, "right_only": c, "p_value": erfc(sqrt(statistic / 2))}
