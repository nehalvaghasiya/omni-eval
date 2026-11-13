def citation_validity(citations: list[str], available: set[str]) -> float:
    """Reference-free structural check, not a truth/faithfulness score."""
    if not citations:
        return 1.0

    return sum(c in available for c in citations) / len(citations)
