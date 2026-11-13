from typing import Any


def state_success(expected: dict[str, Any], observed: dict[str, Any]) -> float:
    """Observed sandbox/tool state, not the agent's generated assertion."""
    return float(
        all(
            key in observed and observed[key] == value
            for key, value in expected.items()
        )
    )
