"""Read-only FastAPI app, loopback only until authenticated server exists."""

from pathlib import Path

from fastapi import FastAPI, HTTPException

from omni_eval.core.registry import Registry
from omni_eval.storage.store import Store


def create_app(runs: str | Path) -> FastAPI:
    """Create a local-only run browser; caller must bind to loopback."""
    app = FastAPI(title="OmniEval local run browser", version="0.2.0", docs_url=None)
    root = Path(runs).resolve()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics")
    def metrics() -> list[str]:
        return Registry().list()

    @app.get("/runs")
    def list_runs() -> list[str]:
        return (
            sorted(p.name for p in root.iterdir() if p.is_dir() and (p / "checksums.json").exists())
            if root.exists()
            else []
        )

    @app.get("/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, object]:
        if not run_id.isalnum():
            raise HTTPException(400, "invalid run ID")
        path = root / run_id
        try:
            return Store().load(path).summary
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(404, "run not found or invalid") from exc

    return app
