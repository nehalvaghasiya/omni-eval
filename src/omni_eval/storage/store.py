"""Immutable atomic run directories, checksummed and retention-aware."""

import json
import tempfile
from pathlib import Path

from omni_eval.io.hash import hash_file
from omni_eval.io.jsonl import Jsonl
from omni_eval.reports.html import Html
from omni_eval.types.run import Run


class Store:
    def save(self, run: Run, out: str | Path) -> Path:
        root = Path(out).resolve()
        root.mkdir(parents=True, exist_ok=True)
        if not run.id.isalnum() or len(run.id) > 90:
            raise ValueError("invalid run id")
        dest = root / run.id
        if dest.exists() or dest.is_symlink():
            raise FileExistsError(f"run {run.id} already exists")
        retain = run.config.get("output", {}).get("retain", True)
        with tempfile.TemporaryDirectory(prefix=".omni-", dir=root) as scratch:
            folder = Path(scratch)
            (folder / "partial").write_text("incomplete", encoding="utf-8")
            manifest = dict(run.manifest)
            manifest["retain_records"] = bool(retain)
            (folder / "manifest.json").write_text(
                json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False), encoding="utf-8"
            )
            (folder / "scores.json").write_text(
                json.dumps(
                    [s.model_dump(mode="json", exclude={"points"}) for s in run.scores],
                    sort_keys=True,
                    indent=2,
                    allow_nan=False,
                ),
                encoding="utf-8",
            )
            (folder / "run.json").write_text(
                json.dumps(
                    {
                        "id": run.id,
                        "status": run.status,
                        "task": run.task,
                        "fixture": run.fixture,
                        "config": run.config,
                    },
                    sort_keys=True,
                    indent=2,
                    allow_nan=False,
                ),
                encoding="utf-8",
            )
            if retain:
                Jsonl().write(
                    folder / "cases.jsonl", [c.model_dump(mode="json") for c in run.cases]
                )
                Jsonl().write(
                    folder / "points.jsonl",
                    [p.model_dump(mode="json") for s in run.scores for p in s.points],
                )
                Jsonl().write(folder / "errors.jsonl", [{"message": x} for x in run.errors])
            (folder / "report.html").write_text(Html().render(run), encoding="utf-8")
            (folder / "partial").unlink()
            (folder / "checksums.json").write_text(
                json.dumps(
                    {p.name: hash_file(p) for p in folder.iterdir() if p.is_file()},
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            folder.rename(dest)
        return dest

    def verify(self, path: str | Path) -> None:
        folder = Path(path)
        if folder.is_symlink() or not folder.is_dir() or (folder / "partial").exists():
            raise ValueError("invalid or incomplete run folder")
        checks = json.loads((folder / "checksums.json").read_text(encoding="utf-8"))
        if set(checks) != set(
            p.name for p in folder.iterdir() if p.is_file() and p.name != "checksums.json"
        ):
            raise ValueError("run artifacts are missing or have unexpected additions")
        for name, expected in checks.items():
            file = folder / name
            if name != Path(name).name or file.is_symlink() or hash_file(file) != expected:
                raise ValueError(f"artifact integrity check failed: {name}")

    def load(self, path: str | Path) -> Run:
        self.verify(path)
        folder = Path(path)
        if not (folder / "run.json").exists() and (folder / "metrics.json").exists():
            from omni_eval.storage.legacy import LegacyStore

            return LegacyStore().load(folder)
        info = json.loads((folder / "run.json").read_text(encoding="utf-8"))
        from omni_eval.types.case import Case
        from omni_eval.types.point import Point
        from omni_eval.types.score import Score

        scores = [
            Score.model_validate(s)
            for s in json.loads((folder / "scores.json").read_text(encoding="utf-8"))
        ]
        cases = (
            [Case.model_validate(c) for c in Jsonl().read(folder / "cases.jsonl")]
            if (folder / "cases.jsonl").exists()
            else []
        )
        if (folder / "points.jsonl").exists():
            by_name = {s.name: s for s in scores}
            for point in Jsonl().read(folder / "points.jsonl"):
                record = Point.model_validate(point)
                by_name[record.metric].points.append(record)
        errors = (
            [x["message"] for x in Jsonl().read(folder / "errors.jsonl")]
            if (folder / "errors.jsonl").exists()
            else []
        )
        return Run(
            **info,
            scores=scores,
            cases=cases,
            errors=errors,
            manifest=json.loads((folder / "manifest.json").read_text(encoding="utf-8")),
            path=folder,
        )
