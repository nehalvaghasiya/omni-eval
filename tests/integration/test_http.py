"""Real loopback HTTP integration, no model, provider credential, or internet."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from omni_eval import evaluate
from omni_eval.io.jsonl import Jsonl
from omni_eval.types.config import Config


class Stub(BaseHTTPRequestHandler):
    def do_POST(self):
        size = int(self.headers.get("Content-Length", "0"))
        data = json.loads(self.rfile.read(size))
        if self.path == "/openai":
            output = {"choices": [{"message": {"content": "hello world"}}]}
        else:
            value = data["payload"]["text"]
            output = {"result": {"label": "A" if value == "a" else "B"}}
        payload = json.dumps(output).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        return None


def test_real_loopback_saved_parity(tmp_path, monkeypatch):
    server = ThreadingHTTPServer(("127.0.0.1", 0), Stub)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        path = tmp_path / "data.jsonl"
        Jsonl().write(
            path,
            [
                {"id": "a", "input": {"text": "a"}, "expected": {"label": "A"}},
                {"id": "b", "input": {"text": "b"}, "expected": {"label": "B"}},
            ],
        )
        from omni_eval.types.target import Target

        monkeypatch.setenv("OMNIEVAL_TEST_URL", f"http://127.0.0.1:{server.server_port}/predict")
        target = Target(
            kind="json",
            url_env="OMNIEVAL_TEST_URL",
            allowed_hosts=["127.0.0.1"],
            allowed_ports=[server.server_port],
            local_http=True,
            request={"payload": "$.input"},
            response={"actual": "$.result"},
        )
        run = evaluate(
            Config(
                task="classification",
                data=str(path),
                source="http",
                target=target,
                metrics=["accuracy", "success_rate", "latency_p95"],
            )
        )
        assert run.metric("accuracy").value == 1
        assert run.metric("success_rate").value == 1
        assert run.metric("latency_p95").details["source"] == "measured_by_omnieval"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_openai_compatible_fixture(tmp_path, monkeypatch):
    server = ThreadingHTTPServer(("127.0.0.1", 0), Stub)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        path = tmp_path / "data.jsonl"
        Jsonl().write(path, [{"id": "x", "input": "say hello", "expected": "hello world"}])
        monkeypatch.setenv("OMNIEVAL_TEST_URL", f"http://127.0.0.1:{server.server_port}/openai")
        from omni_eval.types.target import Target

        target = Target(
            kind="openai",
            model="fixture-model",
            url_env="OMNIEVAL_TEST_URL",
            allowed_hosts=["127.0.0.1"],
            allowed_ports=[server.server_port],
            local_http=True,
        )
        run = evaluate(
            Config(task="text", data=str(path), source="openai", target=target, metrics=["exact"])
        )
        assert run.metric("exact").value == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
