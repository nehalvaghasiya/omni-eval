from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor

from omni_eval.endpoints.safe_http_client import SafeHttpClient
from omni_eval.schemas.eval_sample import EvalSample
from omni_eval.schemas.prediction_record import PredictionRecord
from omni_eval.schemas.target_spec import TargetSpec


class HttpPredictionSource:
    """Bounded concurrent inference preserving original sample IDs."""

    def __init__(self, target: TargetSpec):
        self.target = target

    def acquire(self, samples: Iterable[EvalSample]) -> list[PredictionRecord]:
        records = list(samples)
        client = SafeHttpClient(self.target)

        with ThreadPoolExecutor(max_workers=self.target.max_concurrency) as pool:
            return list(pool.map(client.predict, records))
