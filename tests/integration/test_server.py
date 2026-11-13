from fastapi.testclient import TestClient

from omni_eval.api import evaluate
from omni_eval.server.app import create_app


def test_read_only_api_loads_verified_artifacts(tmp_path, classifier):
    samples, predictions = classifier
    run = evaluate(samples, predictions, ["accuracy"], output_dir=tmp_path)

    with TestClient(create_app(tmp_path)) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/runs").json() == [run.manifest.run_id]
        assert "accuracy" in [metric["id"] for metric in client.get("/metrics").json()]
        detail = client.get(f"/runs/{run.manifest.run_id}")
        assert detail.status_code == 200
        assert detail.json()["metrics"][0]["value"] == run.metrics[0].value
        report = client.get(f"/reports/{run.manifest.run_id}")
        assert report.status_code == 200
        assert "OmniEval" in report.text
        assert client.post("/runs").status_code == 405
        assert client.get("/runs/invalid").status_code == 400
        assert client.get(f"/runs/{'0' * 32}").status_code == 404

        (tmp_path / run.manifest.run_id / "metrics.json").write_text("[]")
        assert client.get(f"/runs/{run.manifest.run_id}").status_code == 404


def test_read_only_api_handles_missing_run_directory(tmp_path):
    with TestClient(create_app(tmp_path / "missing")) as client:
        assert client.get("/runs").json() == []
