from pathlib import Path

import typer

from omni_eval.exceptions import DependencyUnavailable


def serve_command(
    runs: Path = typer.Option(Path(".omni-eval/runs"), "--runs"),
    port: int = typer.Option(8000, "--port"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    if dry_run:
        try:
            from fastapi.testclient import TestClient

            from omni_eval.server.app import create_app

            response = TestClient(create_app(runs)).get("/health")

            if response.status_code != 200:
                raise ValueError("server dry-run health check failed")

            typer.echo(
                "DRY RUN / SYNTHETIC: in-process read-only API tested; no port bound"
            )
        except ImportError:
            # Optional FastAPI must not be a core installation/dry-run prerequisite.
            typer.echo(
                "DRY RUN / SYNTHETIC: server configuration validated; API extra not installed"
            )

        return

    try:
        import uvicorn

        from omni_eval.server.app import create_app
    except ImportError as exc:
        raise DependencyUnavailable(
            "install omni-eval[api] for optional read-only server"
        ) from exc

    uvicorn.run(create_app(runs), host="127.0.0.1", port=port)
