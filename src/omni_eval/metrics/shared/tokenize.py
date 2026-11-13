from omni_eval.metrics.shared.normalize import normalize_text


def tokenize(text: str) -> list[str]:
    """Whitespace tokens after NFKC/casefold/punctuation normalization."""
    return normalize_text(text).split()
