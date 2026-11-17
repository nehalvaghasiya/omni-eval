"""Bounded HTTP acquisition, no redirects, no implicit credentials."""

import os
import time
from typing import Any

import httpx

from omni_eval.sources.base import Source
from omni_eval.sources.policy import HostPolicy
from omni_eval.types.case import Case
from omni_eval.types.config import Config
from omni_eval.types.timing import TimingData


class HttpSource(Source):
    def acquire(self, cases: list[Case], config: Config) -> list[Case]:
        target = config.target
        if target is None:
            raise ValueError("HTTP source requires target")
        url = os.getenv(target.url_env)
        if not url:
            raise ValueError(f"set environment variable {target.url_env}")
        HostPolicy().check_url(
            url,
            target.allowed_hosts,
            local_http=target.local_http,
            allowed_ports=target.allowed_ports,
        )
        token = os.getenv(target.token_env) if target.token_env else None
        headers = {"Accept": "application/json"}
        if token:
            headers["Authorization"] = "Bearer " + token
        result = []
        with httpx.Client(
            timeout=target.timeout, follow_redirects=False, trust_env=False, headers=headers
        ) as client:
            for case in cases:
                started = time.perf_counter()
                observed: Any = None
                error: str | None = None
                attempts = 0
                for attempts in range(target.retries + 1):
                    try:
                        request = self._map(case, target.request)
                        reply = client.post(url, json=request)
                        if reply.is_redirect:
                            raise PermissionError("redirect rejected by endpoint policy")
                        reply.raise_for_status()
                        if len(reply.content) > 2000000:
                            raise ValueError("response exceeds 2MB limit")
                        data = reply.json()
                        observed = self._read(data, target.response.get("actual", "$.actual"))
                        if observed is None:
                            raise ValueError("mapped response field is missing")
                        break
                    except (httpx.HTTPError, ValueError, PermissionError) as exc:
                        error = f"target request failed: {type(exc).__name__}"
                        if attempts >= target.retries or isinstance(exc, PermissionError):
                            break
                elapsed = (time.perf_counter() - started) * 1000
                result.append(
                    case.model_copy(
                        update={
                            "actual": observed,
                            "error": None if observed is not None else error,
                            "timing": TimingData(
                                latency_ms=elapsed, source="measured_by_omnieval", retries=attempts
                            ),
                        }
                    )
                )
        return result

    def _read(self, data: Any, path: str) -> Any:
        if not path.startswith("$.") and path != "$":
            raise ValueError("mapping must be a safe $.field path")
        if path == "$":
            return data
        for part in path[2:].split("."):
            if isinstance(data, dict):
                data = data.get(part)
            elif isinstance(data, list) and part.isdigit():
                n = int(part)
                data = data[n] if n < len(data) else None
            else:
                return None
        return data

    def _map(self, case: Case, mapping: dict[str, str]) -> dict[str, Any]:
        return {key: self._read(case.model_dump(), value) for key, value in mapping.items()} or {
            "input": case.input
        }
