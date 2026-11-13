"""Content-addressed, immutable run directories committed via atomic rename."""

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from omni_eval.exceptions import InputError
from omni_eval.reports.html_reporter import HtmlReporter
from omni_eval.reports.junit_reporter import JunitReporter
from omni_eval.reports.markdown_reporter import MarkdownReporter
from omni_eval.schemas.eval_sample import EvalSample
from omni_eval.schemas.metric_result import MetricResult
from omni_eval.schemas.prediction_record import PredictionRecord
from omni_eval.schemas.run_manifest import RunManifest
from omni_eval.schemas.run_result import RunResult
from omni_eval.schemas.sample_score import SampleScore
from omni_eval.sources.jsonl_loader import write_jsonl


class FileRunRepository:
    """Local filesystem store, safe from accidental overwrite or partial completion."""

    def save(
        self,
        run: RunResult,
        output_dir: str | Path,
        *,
        formats: tuple[str, ...] = ("json", "html"),
        retain_records: bool = True,
    ) -> str:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        destination = root / run.manifest.run_id
        temporary = root / f".{run.manifest.run_id}.partial"

        if destination.exists() or temporary.exists():
            raise InputError("run ID collision; immutable runs cannot be overwritten")

        temporary.mkdir()

        try:
            self._json(
                temporary / "manifest.json", run.manifest.model_dump(mode="json")
            )
            self._json(
                temporary / "metrics.json",
                [m.model_dump(mode="json") for m in run.metrics],
            )

            if retain_records:
                write_jsonl(temporary / "samples.jsonl", run.samples)
                write_jsonl(temporary / "predictions.jsonl", run.predictions)

            write_jsonl(temporary / "sample_scores.jsonl", run.sample_scores)
            (temporary / "errors.jsonl").write_text(
                "".join(json.dumps({"error": err}) + "\n" for err in run.errors),
                encoding="utf-8",
            )

            if "hybrid_comparisons" in run.manifest.metadata:
                self._json(
                    temporary / "comparison.json",
                    run.manifest.metadata["hybrid_comparisons"],
                )

            if "html" in formats:
                (temporary / "report.html").write_text(
                    HtmlReporter().render(run), encoding="utf-8"
                )

            if "markdown" in formats:
                (temporary / "report.md").write_text(
                    MarkdownReporter().render(run), encoding="utf-8"
                )

            if "junit" in formats:
                (temporary / "junit.xml").write_text(
                    JunitReporter().render(run), encoding="utf-8"
                )

            checksums = {
                file.name: hashlib.sha256(file.read_bytes()).hexdigest()
                for file in temporary.iterdir()
                if file.is_file()
            }
            self._json(temporary / "checksums.json", checksums)
            os.replace(temporary, destination)
        except Exception:
            (temporary / "PARTIAL").touch(exist_ok=True)

            raise

        run.path = str(destination)

        return str(destination)

    def load(self, path: str | Path) -> RunResult:
        folder = Path(path)

        if not folder.is_dir() or (folder / "PARTIAL").exists():
            raise InputError(f"not a completed run directory: {folder}")

        try:
            checksums: dict[str, str] = json.loads(
                (folder / "checksums.json").read_text(encoding="utf-8")
            )

            for name, expected in checksums.items():
                if (
                    name != Path(name).name
                    or hashlib.sha256((folder / name).read_bytes()).hexdigest()
                    != expected
                ):
                    raise InputError(f"artifact integrity mismatch: {name}")

            manifest = RunManifest.model_validate_json(
                (folder / "manifest.json").read_text(encoding="utf-8")
            )
            metrics = [
                MetricResult.model_validate(x)
                for x in json.loads(
                    (folder / "metrics.json").read_text(encoding="utf-8")
                )
            ]

            def read_lines(name: str, model: type[Any]) -> list[Any]:
                file = folder / name

                return (
                    [
                        model.model_validate_json(line)
                        for line in file.read_text(encoding="utf-8").splitlines()
                        if line.strip()
                    ]
                    if file.exists()
                    else []
                )

            errors = (
                [
                    json.loads(line)["error"]
                    for line in (folder / "errors.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                    if line.strip()
                ]
                if (folder / "errors.jsonl").exists()
                else []
            )

            return RunResult(
                manifest=manifest,
                metrics=metrics,
                sample_scores=read_lines("sample_scores.jsonl", SampleScore),
                samples=read_lines("samples.jsonl", EvalSample),
                predictions=read_lines("predictions.jsonl", PredictionRecord),
                errors=errors,
                path=str(folder),
            )
        except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
            if isinstance(exc, InputError):
                raise

            raise InputError(f"invalid run artifact {folder}: {exc}") from exc

    @staticmethod
    def _json(path: Path, value: Any) -> None:
        path.write_text(
            json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
