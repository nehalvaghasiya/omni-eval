"""The optional server reads persisted artifacts; no separate scorer."""

import pytest

from omni_eval import evaluate
from omni_eval.fixtures.data import cases


def test_local_read_only_api(tmp_path):
    pytest.importorskip("fastapi", reason="optional api extra is absent")
    from fastapi.testclient import TestClient

    from omni_eval.server.app import create_app

    run = evaluate(cases("classification"), task="classification", out=tmp_path)
    client = TestClient(create_app(tmp_path))
    assert client.get("/health").json() == {"status": "ok"}
    assert run.id in client.get("/runs").json()
    assert (
        client.get(f"/runs/{run.id}").json()["metrics"]["accuracy"] == run.metric("accuracy").value
    )
    assert client.get("/runs/../../etc/passwd").status_code in (404, 400)
    assert client.get("/metrics").json()
    assert client.get("/runs/missing").status_code == 404
    assert client.post("/runs").status_code == 405
