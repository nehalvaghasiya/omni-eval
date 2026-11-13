"""Local read-only run explorer; never schedules side-effecting evaluation jobs."""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from omni_eval.core.metric_registry import MetricRegistry
from omni_eval.schemas.run_result import RunResult
from omni_eval.storage.file_run_repository import FileRunRepository


def create_app(root: str | Path) -> FastAPI:
    app = FastAPI(title="OmniEval read-only local API", docs_url="/docs")
    runs_dir = Path(root).resolve()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics")
    def metrics() -> list[dict[str, object]]:
        return [
            vars(descriptor)
            if hasattr(descriptor, "__dict__")
            else {"id": descriptor.id, "family": descriptor.family}
            for descriptor in MetricRegistry().list()
        ]

    @app.get("/runs")
    def runs() -> list[str]:
        if not runs_dir.exists():
            return []

        return sorted(
            path.name
            for path in runs_dir.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        )

    def verified_run(run_id: str) -> RunResult:
        if not run_id.isalnum() or len(run_id) != 32:
            raise HTTPException(status_code=400, detail="invalid run ID")

        try:
            return FileRunRepository().load(runs_dir / run_id)
        except Exception:
            raise HTTPException(
                status_code=404, detail="run not found or invalid"
            ) from None

    @app.get("/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, object]:
        run = verified_run(run_id)

        return {
            "manifest": run.manifest.model_dump(mode="json"),
            "metrics": [metric.model_dump(mode="json") for metric in run.metrics],
        }

    @app.get("/reports/{run_id}", response_class=HTMLResponse)
    def get_report(run_id: str) -> str:
        from omni_eval.reports.html_reporter import HtmlReporter

        return HtmlReporter().render(verified_run(run_id))

    return app
