"""OpenAI-compatible chat endpoint via the same safe HTTP transport."""

import os
import time

import httpx

from omni_eval.sources.http import HttpSource
from omni_eval.sources.policy import HostPolicy
from omni_eval.types.case import Case
from omni_eval.types.config import Config
from omni_eval.types.timing import TimingData


class OpenAISource(HttpSource):
    def acquire(self, cases: list[Case], config: Config) -> list[Case]:
        target = config.target
        if target is None:
            raise ValueError("OpenAI source requires target")
        url = os.environ.get(target.url_env)
        if not url:
            raise ValueError(f"set {target.url_env}")
        HostPolicy().check_url(
            url,
            target.allowed_hosts,
            local_http=target.local_http,
            allowed_ports=target.allowed_ports,
        )
        headers = {}
        if target.token_env and os.environ.get(target.token_env):
            headers["Authorization"] = "Bearer " + os.environ[target.token_env]
        out = []
        if not target.model:
            raise ValueError("OpenAI-compatible target needs an explicit model")
        with httpx.Client(
            timeout=target.timeout, follow_redirects=False, trust_env=False, headers=headers
        ) as client:
            for case in cases:
                started = time.perf_counter()
                error = None
                actual = None
                attempts = 0
                for attempts in range(target.retries + 1):
                    try:
                        response = client.post(
                            url,
                            json={
                                "model": target.model,
                                "messages": [{"role": "user", "content": str(case.input)}],
                            },
                        )
                        if response.is_redirect:
                            raise PermissionError("redirect refused")
                        response.raise_for_status()
                        if len(response.content) > 2000000:
                            raise ValueError("response exceeds size limit")
                        content = response.json()["choices"][0]["message"]["content"]
                        if not isinstance(content, str):
                            raise ValueError("chat content must be text")
                        actual = content
                        error = None
                        break
                    except (
                        KeyError,
                        IndexError,
                        TypeError,
                        ValueError,
                        httpx.HTTPError,
                        PermissionError,
                    ) as exc:
                        error = f"openai-compatible request failed: {type(exc).__name__}"
                        if attempts >= target.retries or isinstance(exc, PermissionError):
                            break
                out.append(
                    case.model_copy(
                        update={
                            "actual": actual,
                            "error": error,
                            "timing": TimingData(
                                latency_ms=(time.perf_counter() - started) * 1000,
                                source="measured_by_omnieval",
                                retries=attempts,
                            ),
                        }
                    )
                )
        return out
