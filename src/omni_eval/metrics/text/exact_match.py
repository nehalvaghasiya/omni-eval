from omni_eval.metrics.shared.normalize import normalize_text


def exact_match(prediction: str, reference: str, *, normalized: bool = False) -> float:
    if normalized:
        return float(normalize_text(prediction) == normalize_text(reference))

    return float(prediction == reference)
