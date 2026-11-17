"""Supervised small worker protocol: no shell and bounded output/time."""

import os
import subprocess
import sys
from uuid import uuid4

from omni_eval.plugins.reply import WorkerReply
from omni_eval.plugins.request import WorkerRequest
from omni_eval.types.batch import Batch
from omni_eval.types.config import Config
from omni_eval.types.score import Score


class Runner:
    def score(self, name: str, batch: Batch, config: Config) -> list[Score]:
        provider = name.split(":", 1)[0]
        from omni_eval.plugins.catalog import Catalog

        Catalog().require(provider, name)
        if provider == "judge":
            from omni_eval.plugins.judge import JudgePlugin

            return [JudgePlugin().score(name, batch, config)]
        if provider != "fixture":
            raise RuntimeError(f"no qualified worker available for {provider}")
        request = WorkerRequest(
            job=uuid4().hex,
            provider=provider,
            operation="score",
            metrics=[name],
            cases=[c.model_dump(mode="json") for c in batch.cases],
            limits=config.limits,
        )
        command = [sys.executable, "-m", "omni_eval.plugins.worker"]
        env = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": os.pathsep.join(sys.path),
            "PYTHONIOENCODING": "utf-8",
        }
        try:
            completed = subprocess.run(
                command,
                input=request.model_dump_json(),
                text=True,
                capture_output=True,
                timeout=15,
                env=env,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("fixture worker exceeded time budget") from exc
        if completed.returncode or len(completed.stdout) > 1000000:
            raise RuntimeError("worker crashed or exceeded output limit")
        reply = WorkerReply.model_validate_json(completed.stdout)
        if reply.job != request.job or reply.provider != provider:
            raise RuntimeError("worker identity mismatch")
        if reply.status != "complete":
            raise RuntimeError("worker did not complete")
        return [Score.model_validate(item) for item in reply.metrics]
