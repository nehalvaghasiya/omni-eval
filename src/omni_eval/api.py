"""Five easy public entry points; single shared engine."""

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from omni_eval.core.compare import Comparator
from omni_eval.core.engine import Engine
from omni_eval.core.registry import Registry
from omni_eval.fixtures.data import cases as demo_cases
from omni_eval.sources.file import FileSource
from omni_eval.storage.store import Store
from omni_eval.types.case import Case
from omni_eval.types.config import Config
from omni_eval.types.run import Run


def evaluate(
    data: str | Path | Config | Iterable[Case],
    *,
    task: str | None = None,
    predictions: str | None = None,
    metrics: list[str] | None = None,
    out: str | Path | None = None,
    dry_run: bool = False,
) -> Run:
    """Evaluate one JSONL file, YAML configuration, typed config or iterable cases.

    Invalid input raises ValueError; unprepared plugins raise RuntimeError.
    With dry_run=True all data/model access is replaced by synthetic cases.
    """
    if isinstance(data, Config):
        config = data
    elif isinstance(data, (str, Path)) and str(data).endswith((".yaml", ".yml")):
        config = Config.from_file(data)
    elif isinstance(data, (str, Path)):
        if not task:
            raise ValueError("task is required for a JSONL data file")
        config = Config(task=task, data=str(data), predictions=predictions, metrics=metrics or [])
    else:
        if not task:
            raise ValueError("task is required for iterable cases")
        config = Config(task=task, data="<in-memory>", metrics=metrics or [])
    if metrics is not None:
        config = config.model_copy(update={"metrics": metrics})
    if predictions is not None:
        config = config.model_copy(update={"predictions": predictions})
    if config.plugin and (not dry_run):
        from omni_eval.plugins.catalog import Catalog

        Catalog().require(config.plugin)
        raise RuntimeError(
            f"suite execution for {config.plugin!r} requires a qualified isolated worker"
        )
    if dry_run:
        fixture_task = (
            config.task
            if config.task
            in {
                "classification",
                "text",
                "retrieval",
                "rag",
                "agent",
                "mcp",
                "audio",
                "vision",
                "security",
                "code",
                "sql",
            }
            else "classification"
        )
        source = demo_cases(fixture_task)
        if config.plugin or any(":" in name for name in config.metrics):
            config = config.model_copy(update={"metrics": ["fixture:echo"]})
        elif config.metrics:
            # A dry run must exercise the requested metric, never quietly select
            # a different one. Dynamic names such as recall@5 are supported.
            from omni_eval.core.planner import Planner

            Planner().build(config, source)
        if not config.metrics:
            defaults = {
                "security": ["attack_success"],
                "code": ["code_valid"],
                "sql": ["sql_select_safe"],
            }
            config = config.model_copy(update={"metrics": defaults.get(fixture_task, [])})
        result = Engine().run(source, config, fixture=True)
    else:
        if isinstance(data, (str, Path, Config)):
            if config.source == "replay":
                from omni_eval.sources.replay import ReplaySource

                source = ReplaySource().acquire([], config)
            else:
                files = FileSource()
                if config.source in {"http", "openai"}:
                    if not config.data:
                        raise ValueError("live target requires dataset")
                    raw = Config.model_validate({**config.model_dump(), "predictions": None})
                    source = files.read(raw)
                    from omni_eval.core.planner import Planner

                    Planner().build(config, source)
                    if config.source == "http":
                        from omni_eval.sources.http import HttpSource

                        source = HttpSource().acquire(source, config)
                    else:
                        from omni_eval.sources.openai import OpenAISource

                        source = OpenAISource().acquire(source, config)
                else:
                    source = files.read(config)
        else:
            source = [Case.model_validate(c) for c in data]
        result = Engine().run(source, config)
    target = out or config.output.get("dir")
    if target:
        path = Store().save(result, target)
        result.path = path
    return result


def load(path: str | Path) -> Run:
    return Store().load(path)


def compare(
    baseline: Run,
    candidate: Run,
    *,
    metrics: list[str] | None = None,
    seed: int = 42,
    confidence: float = 0.95,
) -> Any:
    return Comparator().compare(baseline, candidate, metrics, seed=seed, confidence=confidence)


def metrics(task: str | None = None) -> list[str]:
    return Registry().list(task)
