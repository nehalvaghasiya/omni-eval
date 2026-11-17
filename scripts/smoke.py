"""Known-answer examples, persisted artifacts, and optional 10K-case diagnostics."""

import argparse
import json
import math
import tempfile
import time
import tracemalloc
import xml.etree.ElementTree as ET
from pathlib import Path

from omni_eval import evaluate, load
from omni_eval.fixtures.data import cases
from omni_eval.reports.html import Html
from omni_eval.reports.json import Json
from omni_eval.reports.junit import Junit
from omni_eval.reports.md import Markdown

EXPECTED = {
    "classification": {"accuracy": 2 / 3, "macro_f1": 2 / 3},
    "text": {"exact": 0, "token_f1": 1},
    "retrieval": {"hit@5": 1, "recall@5": 0.5, "mrr": 1},
    "rag": {"answer_match": 1, "citation_valid": 1},
    "agent": {"state_success": 1, "tool_allowed": 1},
    "mcp": {"state_success": 1, "tool_allowed": 1},
    "audio": {"wer": 0.5, "cer": 1 / 11},
    "vision": {"accuracy": 1},
    "security": {"attack_success": 0},
    "code": {"code_valid": 1},
    "sql": {"sql_select_safe": 1},
}


def benchmark() -> None:
    for task in ("classification", "text", "retrieval"):
        tracemalloc.start()
        started = time.perf_counter()
        sample = cases(task)
        data = [
            sample[i % len(sample)].model_copy(deep=True, update={"id": str(i)})
            for i in range(10_000)
        ]
        run = evaluate(data, task=task)
        assert run.status == "complete"
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print(
            json.dumps(
                {
                    "task": task,
                    "cases": len(data),
                    "seconds": round(time.perf_counter() - started, 3),
                    "peak_python_mib": round(peak / 1024**2, 2),
                }
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--performance", action="store_true", help="also measure three synthetic 10K-case tasks"
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "examples").glob("*/eval.yaml"))
    assert {path.parent.name for path in paths} == set(EXPECTED)
    with tempfile.TemporaryDirectory(prefix="omnieval-smoke-") as out:
        for path in paths:
            run = evaluate(path, out=out)
            assert run.status == "complete", path
            assert {s.name for s in run.scores} == set(EXPECTED[path.parent.name])
            for score in run.scores:
                assert score.status == "complete" and score.coverage == 1, (path, score)
                expected = EXPECTED[path.parent.name][score.name]
                assert score.value is not None and math.isclose(
                    score.value, expected, rel_tol=1e-9, abs_tol=1e-12
                ), (path, score)
            restored = load(run.path)
            assert restored.summary == run.summary
            assert Json().render(restored)["scores"]
            assert "<table>" in Html().render(restored)
            assert "| Metric |" in Markdown().render(restored)
            assert ET.fromstring(Junit().render(restored)).tag == "testsuite"
            synthetic = evaluate(path, dry_run=True)
            assert synthetic.fixture and synthetic.status == "complete"
            print(
                f"PASS {path.parent.name}: known scores, full coverage, save/load, four reports, dry-run"
            )
    if args.performance:
        benchmark()


if __name__ == "__main__":
    main()
