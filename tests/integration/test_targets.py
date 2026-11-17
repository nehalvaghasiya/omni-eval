"""Bounded fake HTTP responses, retries, error accounting, and redaction."""

import httpx
import pytest

from omni_eval.core.engine import Engine
from omni_eval.sources.http import HttpSource
from omni_eval.sources.openai import OpenAISource
from omni_eval.types.case import Case
from omni_eval.types.config import Config
from omni_eval.types.target import Target


@pytest.fixture
def transport(monkeypatch):
    original = httpx.Client
    monkeypatch.setenv("OMNIEVAL_FAKE_URL", "http://127.0.0.1/predict")
    monkeypatch.setenv("OMNIEVAL_FAKE_TOKEN", "synthetic-secret-never-log")

    def install(handler):
        def client(**kwargs):
            assert kwargs["follow_redirects"] is False
            assert kwargs["trust_env"] is False
            return original(transport=httpx.MockTransport(handler), **kwargs)

        monkeypatch.setattr(httpx, "Client", client)

    return install


def config(source="http", retries=0):
    return Config(
        task="text",
        data="unused",
        source=source,
        metrics=["exact", "success_rate"],
        target=Target(
            url_env="OMNIEVAL_FAKE_URL",
            token_env="OMNIEVAL_FAKE_TOKEN",
            allowed_hosts=["127.0.0.1"],
            allowed_ports=[80],
            local_http=True,
            model="fake",
            retries=retries,
        ),
    )


@pytest.mark.parametrize("source,klass", [("http", HttpSource), ("openai", OpenAISource)])
def test_retry_recovers_and_records_attempts(source, klass, transport):
    calls = []

    def handler(request):
        calls.append(request)
        assert request.headers["Authorization"] == "Bearer synthetic-secret-never-log"
        if len(calls) == 1:
            return httpx.Response(503)
        return httpx.Response(
            200, json={"actual": "ok", "choices": [{"message": {"content": "ok"}}]}
        )

    transport(handler)
    cfg = config(source, retries=1)
    acquired = klass().acquire([Case(id="x", input="q", expected="ok")], cfg)
    assert len(calls) == 2
    assert acquired[0].actual == "ok"
    assert acquired[0].error is None
    assert acquired[0].timing.retries == 1
    assert Engine().run(acquired, cfg).metric("success_rate").value == 1


@pytest.mark.parametrize("source,klass", [("http", HttpSource), ("openai", OpenAISource)])
@pytest.mark.parametrize("failure", ["timeout", "redirect", "malformed", "missing", "oversized"])
def test_target_failures_remain_visible(source, klass, failure, transport):
    def handler(request):
        if failure == "timeout":
            raise httpx.ReadTimeout("synthetic-secret-never-log", request=request)
        if failure == "redirect":
            return httpx.Response(302, headers={"location": "http://external.invalid"})
        if failure == "malformed":
            return httpx.Response(200, content=b"not-json")
        if failure == "missing":
            return httpx.Response(200, json={})
        return httpx.Response(200, content=b"x" * 2_000_001)

    transport(handler)
    cfg = config(source)
    acquired = klass().acquire([Case(id="x", expected="ok")], cfg)
    assert acquired[0].actual is None
    assert acquired[0].error
    run = Engine().run(acquired, cfg)
    assert run.metric("exact").value is None
    assert run.metric("exact").n_failed == 1
    assert run.metric("success_rate").value == 0
    assert "synthetic-secret-never-log" not in run.model_dump_json()


def test_invalid_references_are_rejected_before_transport(tmp_path, transport):
    from omni_eval import evaluate
    from omni_eval.io.jsonl import Jsonl

    def forbidden(request):
        raise AssertionError("transport called before reference validation")

    transport(forbidden)
    path = tmp_path / "cases.jsonl"
    Jsonl().write(path, [{"id": "x", "input": "q"}])
    cfg = config().model_copy(update={"data": str(path)})
    with pytest.raises(ValueError, match="required references"):
        evaluate(cfg)
