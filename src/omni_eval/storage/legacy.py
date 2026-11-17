"""Read old schema-1.0 OmniEval artifacts without rewriting or rerunning them."""

import json
from pathlib import Path

from omni_eval.core.align import Aligner
from omni_eval.io.jsonl import Jsonl
from omni_eval.io.mapping import Mapper
from omni_eval.types.point import Point
from omni_eval.types.run import Run
from omni_eval.types.score import Score


class LegacyStore:
    """Convert the previous repository's saved run shape into the current Run API."""

    def load(self, path: str | Path) -> Run:
        folder = Path(path)
        from omni_eval.storage.store import Store

        Store().verify(folder)
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("schema_version") != "1.0":
            raise ValueError("legacy reader only supports schema version 1.0")
        rows = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
        scores = []
        for row in rows:
            status = {
                "completed": "complete",
                "insufficient_reference": "missing_reference",
                "insufficient_input": "missing_input",
            }.get(row.get("status"), row.get("status", "error"))
            scores.append(
                Score(
                    name=row["name"],
                    value=row.get("value"),
                    status=status,
                    n_total=row.get("n_total", row.get("count", 0)),
                    n_eligible=row.get("n_eligible", row.get("count", 0)),
                    n_scored=row.get("n_scored", row.get("count", 0)),
                    n_failed=row.get("n_failed", 0),
                    n_missing=row.get("n_missing", 0),
                    details=row.get("details", {}),
                    version=str(row.get("details", {}).get("metric_version", "1")),
                    source="legacy-native",
                )
            )
        mapped = (
            [Mapper().to_case(x) for x in Jsonl().read(folder / "samples.jsonl")]
            if (folder / "samples.jsonl").exists()
            else []
        )
        predictions = (
            list(Jsonl().read(folder / "predictions.jsonl"))
            if (folder / "predictions.jsonl").exists()
            else []
        )
        if mapped and predictions:
            mapped = Aligner().join(mapped, predictions)
        if (folder / "sample_scores.jsonl").exists():
            by_name = {s.name: s for s in scores}
            for raw in Jsonl().read(folder / "sample_scores.jsonl"):
                if raw["metric"] not in by_name:
                    continue
                status = (
                    "complete" if raw.get("status") == "completed" else raw.get("status", "error")
                )
                by_name[raw["metric"]].points.append(
                    Point(
                        id=raw["sample_id"],
                        metric=raw["metric"],
                        value=raw.get("value"),
                        status=status,
                        reason=raw.get("reason"),
                    )
                )
        errors = (
            [row.get("error", "") for row in Jsonl().read(folder / "errors.jsonl")]
            if (folder / "errors.jsonl").exists()
            else []
        )
        return Run(
            id=manifest["run_id"],
            status="complete" if manifest.get("status") == "completed" else "partial",
            task=str(manifest.get("metadata", {}).get("task", "legacy")),
            scores=scores,
            cases=mapped,
            fixture=bool(manifest.get("is_fixture", False)),
            manifest=manifest,
            config={"version": 1, "source": "legacy"},
            errors=errors,
            path=folder,
        )
