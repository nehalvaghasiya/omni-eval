from collections.abc import Sequence


def edit_distance(left: Sequence[str] | str, right: Sequence[str] | str) -> int:
    previous = list(range(len(right) + 1))

    for i, left_token in enumerate(left, 1):
        current = [i]

        for j, right_token in enumerate(right, 1):
            current.append(
                min(
                    current[j - 1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (left_token != right_token),
                )
            )

        previous = current

    return previous[-1]


def error_rate(prediction: str, reference: str, *, characters: bool = False) -> float:
    """Levenshtein errors / reference length; may exceed 1 for insertions."""
    left = list(prediction) if characters else prediction.split()
    right = list(reference) if characters else reference.split()

    if not right:
        return 0.0 if not left else float(len(left))

    return edit_distance(left, right) / len(right)
