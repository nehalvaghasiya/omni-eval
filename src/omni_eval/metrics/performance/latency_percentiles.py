import numpy as np


def latency_percentile(values: list[float], quantile: float) -> float:
    if not values or not 0 <= quantile <= 100:
        raise ValueError(
            "latency percentile requires observations and quantile [0,100]"
        )

    return float(np.percentile(values, quantile, method="linear"))
