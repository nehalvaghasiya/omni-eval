"""Per-run hard call and declared-token budget."""


class Budget:
    """Refuse a judge call once either finite quota is exhausted."""

    def __init__(self, max_calls: int, max_tokens: int):
        if max_calls <= 0 or max_tokens <= 0:
            raise ValueError("judge budgets must be positive")
        self.max_calls = max_calls
        self.max_tokens = max_tokens
        self.calls = 0
        self.tokens = 0

    def reserve(self, estimated_tokens: int) -> None:
        if self.calls >= self.max_calls or self.tokens + estimated_tokens > self.max_tokens:
            raise RuntimeError("judge call/token budget exceeded")
        self.calls += 1
        self.tokens += estimated_tokens
