import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

from omni_eval.api import evaluate
from omni_eval.endpoints.endpoint_policy import EndpointPolicy
from omni_eval.exceptions import SecurityError
from omni_eval.schemas.target_spec import TargetSpec
from omni_eval.sources.http_prediction_source import HttpPredictionSource


def test_endpoint_policy_blocks_metadata():
    target = TargetSpec(
        url="http://169.254.169.254/latest/meta-data",
        allowed_hosts=["169.254.169.254"],
        allow_localhost=True,
    )

    with pytest.raises(SecurityError):
        EndpointPolicy().validate(target.url, target)


def test_endpoint_policy_blocks_unapproved_host():
    target = TargetSpec(
        url="https://not-approved.invalid/model", allowed_hosts=["approved.example"]
    )

    with pytest.raises(SecurityError):
        EndpointPolicy().validate(target.url, target)


def test_local_http_equals_offline(classifier):
    samples, predictions = classifier
    answers = {p.sample_id: p for p in predictions}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            size = int(self.headers["Content-Length"])
            payload = json.loads(self.rfile.read(size))
            sid = payload["sample_id"]
            result = {
                "result": answers[sid].output,
                "probabilities": answers[sid].probabilities.model_dump(),
            }
            raw = json.dumps(result).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()

    try:
        port = server.server_port
        target = TargetSpec(
            url=f"http://127.0.0.1:{port}/infer",
            allowed_hosts=[f"127.0.0.1:{port}"],
            allow_localhost=True,
            request_template={"sample_id": "$.sample_id"},
            response_map={"output": "$.result", "probabilities": "$.probabilities"},
        )
        live = HttpPredictionSource(target).acquire(samples)

        assert all(
            p.timing is not None and p.timing.provenance == "measured_by_omnieval"
            for p in live
        )
        offline_run = evaluate(
            samples, predictions, ["accuracy", "macro_f1", "brier_score"]
        )
        online_run = evaluate(samples, live, ["accuracy", "macro_f1", "brier_score"])

        assert [m.value for m in offline_run.metrics] == [
            m.value for m in online_run.metrics
        ]
        assert evaluate(samples, live, ["latency_p95"]).metrics[0].value > 0
    finally:
        server.shutdown()
        server.server_close()
