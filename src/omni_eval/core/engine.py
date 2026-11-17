"""Single scoring engine, independent of file and model acquisition."""

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from omni_eval.core.planner import Planner
from omni_eval.core.registry import Registry
from omni_eval.io.hash import digest
from omni_eval.types.batch import Batch
from omni_eval.types.case import Case
from omni_eval.types.config import Config
from omni_eval.types.run import Run


class Engine:
    def run(self, cases: list[Case], config: Config, *, fixture: bool = False) -> Run:
        if not cases:
            raise ValueError("dataset contains zero cases")
        if len({case.id for case in cases}) != len(cases):
            raise ValueError("duplicate case id in dataset")
        names = Planner().build(config, cases)
        batch = Batch(task=config.task, cases=cases)
        results = []
        registry = Registry()
        for name in names:
            if ":" in name:
                from omni_eval.plugins.runner import Runner

                results.extend(Runner().score(name, batch, config))
            else:
                metric = registry.get(name)
                settings = config.limits.get("metric_settings", {}).get(name, {})
                score = metric.compute(batch, settings)
                if config.slices:
                    from omni_eval.core.slices import Slicer

                    keys = config.slices.get("keys", [])
                    if not isinstance(keys, list) or any(not isinstance(key, str) for key in keys):
                        raise ValueError("slices.keys must be a list of metadata names")
                    minimum = config.slices.get("min_count", 5)
                    if not isinstance(minimum, int) or minimum < 2:
                        raise ValueError("slices.min_count must be at least two")
                    group_scores: dict[str, dict[str, Any]] = {}
                    for key in keys:
                        groups = Slicer().group(batch, key, min_count=minimum)
                        group_scores[key] = {}
                        for label, group in groups.items():
                            sliced = metric.compute(group, settings)
                            group_scores[key][label] = {
                                "value": sliced.value,
                                "status": sliced.status,
                                "coverage": sliced.coverage,
                                "n_total": sliced.n_total,
                            }
                    score.details["slices"] = group_scores
                results.append(score)
        fixture = fixture or any(score.source == "fixture" for score in results)
        status = "complete" if all(score.status == "complete" for score in results) else "partial"
        clean = config.model_dump(mode="json", exclude_none=True)
        manifest = {
            "schema_version": 1,
            "created": datetime.now(UTC).isoformat(),
            "data_hash": digest(
                [c.model_dump(mode="json") for c in sorted(cases, key=lambda x: x.id)]
            ),
            "config_hash": digest(clean),
            "metric_versions": {s.name: s.version for s in results},
            "metric_sources": {s.name: s.source for s in results},
            "synthetic": fixture,
        }
        return Run(
            id=uuid4().hex,
            status=status,
            task=config.task,
            scores=results,
            config=clean,
            manifest=manifest,
            cases=cases,
            fixture=fixture,
        )
