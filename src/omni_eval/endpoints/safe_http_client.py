"""Safe request/response boundaries; model output remains untrusted data."""

import json
import os
import time
from typing import Any

import httpx

from omni_eval.endpoints.endpoint_policy import EndpointPolicy
from omni_eval.exceptions import AcquisitionError, InputError
from omni_eval.schemas.eval_sample import EvalSample
from omni_eval.schemas.prediction_record import PredictionRecord
from omni_eval.schemas.target_spec import TargetSpec
from omni_eval.sources.field_mapper import get_path, map_template


class SafeHttpClient:
    """Use short-lived connections, no proxies/redirects, bounded response payloads."""

    def __init__(self, spec: TargetSpec):
        self.spec = spec
        self.url = spec.url or (os.environ.get(spec.url_env or "") or "")

        if not self.url:
            raise InputError("target URL environment variable is unset")

        EndpointPolicy().validate(self.url, spec)
        self.token = (
            os.environ.get(spec.authorization_env, "") if spec.authorization_env else ""
        )

    def predict(self, sample: EvalSample) -> PredictionRecord:
        # Validate destination again before each connection; never follow redirects.
        EndpointPolicy().validate(self.url, self.spec)
        template = self.spec.request_template or {"input": "$.input"}
        payload = map_template(template, sample.model_dump(mode="json"))
        headers = {"Accept": "application/json", "Content-Type": "application/json"}

        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        for attempt in range(self.spec.retries + 1):
            try:
                with httpx.Client(
                    timeout=self.spec.timeout_seconds,
                    follow_redirects=False,
                    trust_env=False,
                ) as client:
                    start = time.perf_counter()

                    with client.stream(
                        "POST", self.url, json=payload, headers=headers
                    ) as response:
                        if response.is_redirect:
                            raise AcquisitionError(
                                "endpoint redirect blocked by security policy"
                            )

                        if response.status_code >= 400:
                            raise AcquisitionError(
                                f"endpoint HTTP {response.status_code}"
                            )

                        content = bytearray()

                        for fragment in response.iter_bytes():
                            content.extend(fragment)

                            if len(content) > self.spec.max_response_bytes:
                                raise AcquisitionError(
                                    "endpoint response exceeds size limit"
                                )

                    latency_ms = (time.perf_counter() - start) * 1000

                decoded: Any = json.loads(content)
                response_map = self.spec.response_map

                if self.spec.type == "openai_compatible" and response_map == {
                    "output": "$.result"
                }:
                    response_map = {"output": "$.choices.0.message.content"}

                if "output" not in response_map:
                    raise InputError("response_map requires output selector")

                unknown = set(response_map) - {
                    "output",
                    "probabilities",
                    "contexts",
                    "trace",
                    "rag_trace",
                }

                if unknown:
                    raise InputError(
                        f"unsupported response mapping keys: {sorted(unknown)}"
                    )

                mapped = {
                    key: get_path(decoded, path) for key, path in response_map.items()
                }
                output = mapped.pop("output")

                if not isinstance(output, dict):
                    output = {"text": output}

                return PredictionRecord(
                    sample_id=sample.sample_id,
                    output=output,
                    timing={
                        "latency_ms": latency_ms,
                        "provenance": "measured_by_omnieval",
                    },
                    **mapped,
                )
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                error = (
                    f"endpoint {type(exc).__name__}"  # never include URL/header/secret
                )
            except (ValueError, TypeError) as exc:
                error = f"endpoint invalid JSON/schema: {type(exc).__name__}"
            except (AcquisitionError, InputError) as exc:
                error = str(exc)

                if isinstance(exc, InputError):
                    break

                if "redirect" in error or "size limit" in error or "HTTP 4" in error:
                    break

            if attempt == self.spec.retries:
                break

        return PredictionRecord(sample_id=sample.sample_id, output={}, error=error)
