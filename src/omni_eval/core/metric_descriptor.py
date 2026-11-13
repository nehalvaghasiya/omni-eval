from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class MetricDescriptor:
    """Static capability metadata; this class has no runtime registration side effect."""

    id: str
    family: Literal[
        "classification", "text", "retrieval", "rag", "agent", "performance", "security"
    ]
    required_reference: str | None
    required_prediction: str
    orientation: Literal["higher_is_better", "lower_is_better", "informational"]
    version: str = "1.0"
    score_range: tuple[float, float] | None = (0, 1)
    aggregation: str = "mean"
    dependency_group: str = "core"
    judge_required: bool = False
