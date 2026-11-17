"""Local fake judge tests the genuine OpenAI-compatible HTTP and JSON contracts."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from omni_eval.types.case import Case
from omni_eval.types.config import Config


class JudgeStub(BaseHTTPRequestHandler):
    answer = {
        "score": 0.75,
        "status": "complete",
        "reason": "relevant",
        "revision": "mock-1",
        "tokens": 16,
    }
    messages = None

    def do_POST(self) -> None:
        length = int(self.headers["Content-Length"])
        request = json.loads(self.rfile.read(length))
        JudgeStub.messages = request["messages"]
        body = json.dumps({"choices": [{"message": {"content": json.dumps(self.answer)}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: object) -> None:
        pass


@pytest.fixture
def judge_config(tmp_path, monkeypatch):
    server = ThreadingHTTPServer(("127.0.0.1", 0), JudgeStub)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    rubric = tmp_path / "rubric.yaml"
    rubric.write_text(
        'id: relevance\nversion: "1"\ncriterion: Does the answer address the question?\n'
        'anchors: {"0": "irrelevant", "1": "relevant"}\nrequired: [input, actual]\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("OMNIEVAL_JUDGE_URL", f"http://127.0.0.1:{server.server_port}/chat")
    spec = Config(
        task="text",
        data="in-memory",
        metrics=["judge:relevance"],
        judge={
            "provider": "openai_compatible",
            "model": "mock-judge",
            "rubric": str(rubric),
            "url_env": "OMNIEVAL_JUDGE_URL",
            "allowed_hosts": ["127.0.0.1"],
            "allowed_ports": [server.server_port],
            "local_http": True,
            "max_calls": 3,
            "max_tokens": 3072,
            "per_call_tokens": 1024,
            "approve_external_data": True,
        },
    )
    try:
        yield spec
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)


def test_real_http_judge_response_is_not_ground_truth(judge_config):
    from omni_eval.core.engine import Engine

    actual_run = Engine().run(
        [Case(id="a", input="refund period?", actual="30 days")], judge_config
    )
    score = actual_run.metric("judge:relevance")
    assert score.value == 0.75
    assert score.source == "judge"
    assert score.details["method"].startswith("LLM-as-judge")
    assert score.points[0].reason == "relevant"
    assert JudgeStub.messages[0]["role"] == "system"
    assert JudgeStub.messages[1]["role"] == "user"


def test_missing_reference_field_is_not_fake_zero(judge_config):
    from omni_eval.core.engine import Engine

    run = Engine().run([Case(id="a", input="question")], judge_config)
    score = run.metric("judge:relevance")
    assert score.value is None
    assert score.n_missing == 1


def test_explicit_consent_required(judge_config):
    from omni_eval.core.engine import Engine

    config = judge_config.model_copy(
        update={"judge": {**judge_config.judge, "approve_external_data": False}}
    )
    with pytest.raises(PermissionError, match="approve_external_data"):
        Engine().run([Case(id="a", input="question", actual="answer")], config)


def test_out_of_range_verdict_is_error_not_zero(judge_config):
    from omni_eval.core.engine import Engine

    JudgeStub.answer = {
        "score": 5.0,
        "status": "complete",
        "reason": "invalid",
        "revision": "mock-1",
    }
    try:
        result = Engine().run([Case(id="a", input="question", actual="answer")], judge_config)
        assert result.metric("judge:relevance").value is None
        assert result.metric("judge:relevance").n_failed == 1
    finally:
        JudgeStub.answer = {
            "score": 0.75,
            "status": "complete",
            "reason": "relevant",
            "revision": "mock-1",
            "tokens": 16,
        }


def test_budget_hit_has_partial_coverage(judge_config):
    from omni_eval.core.engine import Engine

    cfg = judge_config.model_copy(update={"judge": {**judge_config.judge, "max_calls": 1}})
    run = Engine().run(
        [
            Case(id="a", input="first", actual="answer"),
            Case(id="b", input="second", actual="answer"),
        ],
        cfg,
    )
    score = run.metric("judge:relevance")
    assert score.value == 0.75
    assert score.n_failed == 1
    assert score.coverage == 0.5


def test_judge_evidence_references_are_verified(judge_config, tmp_path):
    from omni_eval.core.engine import Engine
    from omni_eval.types.context import Context

    rubric = tmp_path / "evidence.yaml"
    rubric.write_text(
        'id: faithfulness\nversion: "1"\ncriterion: Does the context support the answer?\n'
        'anchors: {"0": "no", "1": "yes"}\nrequired: [actual, contexts]\n'
        "require_evidence: true\n",
        encoding="utf-8",
    )
    cfg = judge_config.model_copy(
        update={
            "metrics": ["judge:faithfulness"],
            "judge": {**judge_config.judge, "rubric": str(rubric)},
        }
    )
    case = Case(
        id="f",
        input="question",
        actual="answer",
        contexts=[Context(id="source-1", rank=1, text="evidence")],
    )
    JudgeStub.answer = {
        "score": 0.9,
        "status": "complete",
        "reason": "supported",
        "revision": "mock-1",
        "evidence": ["nonexistent"],
    }
    try:
        run = Engine().run([case], cfg)
        assert run.metric("judge:faithfulness").value is None
        assert run.metric("judge:faithfulness").n_failed == 1
        JudgeStub.answer["evidence"] = ["source-1"]
        run = Engine().run([case], cfg)
        assert run.metric("judge:faithfulness").value == pytest.approx(0.9)
        assert run.metric("judge:faithfulness").points[0].evidence == ["source-1"]
        assert run.metric("judge:faithfulness").details["judge_revisions"] == ["mock-1"]
    finally:
        JudgeStub.answer = {
            "score": 0.75,
            "status": "complete",
            "reason": "relevant",
            "revision": "mock-1",
            "tokens": 16,
        }


def test_judge_rubric_path_resolves_relative_to_config_file(tmp_path):
    from omni_eval.types.config import Config

    config_file = tmp_path / "eval.yaml"
    config_file.write_text(
        "version: 1\ntask: text\ndata: cases.jsonl\n"
        "metrics: [judge:relevance]\njudge:\n"
        "  rubric: rubric.yaml\n",
        encoding="utf-8",
    )
    parsed = Config.from_file(config_file)
    assert parsed.judge["rubric"] == str(tmp_path / "rubric.yaml")
