"""Every advertised native metric has a hand-constructed valid dry-run path."""

from math import exp, log, log2

import pytest

from omni_eval.core.registry import Registry
from omni_eval.fixtures.data import cases
from omni_eval.types.batch import Batch
from omni_eval.types.case import Case

SAMPLES = {
    "classification": cases("classification"),
    "text": cases("text"),
    "retrieval": cases("retrieval"),
    "rag": cases("rag"),
    "agent": cases("agent"),
    "mcp": cases("mcp"),
    "audio": cases("audio"),
    "vision": cases("vision"),
    "security": cases("security"),
    "code": cases("code"),
    "sql": cases("sql"),
}

SETTINGS = {"positive_label": "B", "bins": 10, "schema": {"type": "object"}, "pattern": ".*"}

# Independent answers for the bundled cases, not values copied from a scorer.
GOLD = {
    "accuracy": 2 / 3,
    "macro_f1": 2 / 3,
    "micro_f1": 2 / 3,
    "weighted_f1": 2 / 3,
    "precision": 1 / 2,
    "macro_precision": 3 / 4,
    "micro_precision": 2 / 3,
    "weighted_precision": 5 / 6,
    "recall": 1,
    "macro_recall": 3 / 4,
    "micro_recall": 2 / 3,
    "weighted_recall": 2 / 3,
    "balanced_accuracy": 3 / 4,
    "mcc": 1 / 2,
    "kappa": 2 / 5,
    "confusion": None,
    "brier": (0.02 + 1.28 + 0.32) / 3,
    "logloss": -(log(0.9) + log(0.2) + log(0.6)) / 3,
    "ece": (0.1 + 0.8 + 0.4) / 3,
    "mce": 0.8,
    "topk@k": 2 / 3,
    "auroc": 1 / 2,
    "auprc": 1 / 2,
    "exact": 0,
    "answer_match": 1,
    "token_f1": 1,
    "token_precision": 1,
    "token_recall": 1,
    "json_valid": 0,
    "schema_valid": 1,
    "fields_f1": 1,
    "format": 1,
    "wer": 1 / 2,
    "cer": 1 / 11,
    "nll": 0.5,
    "perplexity": exp(0.5),
    "code_valid": 1,
    "sql_select_safe": 1,
    "precision@k": 1 / 5,
    "recall@k": 1 / 2,
    "hit@k": 1,
    "success@k": 1,
    "mrr": 1,
    "map": 1 / 2,
    "ndcg@k": 1 / (1 + 1 / log2(3)),
    "rprec": 1 / 2,
    "citation_valid": 1,
    "evidence_coverage": 1,
    "context_precision": 1,
    "rerank_delta": (1 - 1 / log2(3)) / (1 + 1 / log2(3)),
    "state_success": 1,
    "state_partial": 1,
    "tool_allowed": 1,
    "tool_success": 1,
    "loop_rate": 0,
    "step_order": 1,
    "retry_rate": 0,
    "attack_success": 0,
    "policy_violation": 0,
    "latency_mean": 10,
    "latency_p50": 10,
    "latency_p95": 10,
    "latency_p99": 10,
    "success_rate": 1,
}


def test_every_registered_metric_has_an_independent_answer():
    assert set(Registry().list()) == set(GOLD)


def prepare(name: str, task: str) -> list[Case]:
    data = [c.model_copy(deep=True) for c in SAMPLES[task]]
    if name in {"latency_mean", "latency_p50", "latency_p95", "latency_p99"}:
        from omni_eval.types.timing import TimingData

        for c in data:
            c.timing = TimingData(latency_ms=10.0, scope="end_to_end", source="measured_external")
    if name in {"nll", "perplexity"}:
        data = [Case(id="x", actual={"logprobs": [-0.5, -0.5]})]
    if name in {"fields_f1", "schema_valid"}:
        data = [Case(id="x", actual={"a": 1}, expected={"a": 1})]
    if name == "step_order":
        for c in data:
            c.expected["required_tools"] = [c.trace.steps[0].name]
    if name == "rerank_delta":
        for c in data:
            c.extensions["before"] = ["d2", "d1"]
    if name == "policy_violation":
        data = [Case(id="x", actual={"policy_result": {"verified": True, "violated": False}})]
    return data


@pytest.mark.parametrize("name", Registry().list())
def test_every_metric_executes(name):
    reg = Registry()
    metric = reg.get(name)
    task = next(
        (
            task
            for task in SAMPLES
            if task in metric.tasks and not (name in {"wer", "cer"} and task != "audio")
        ),
        None,
    )
    assert task is not None
    data = prepare(name, task)
    score = metric.compute(Batch(task=task, cases=data), SETTINGS)
    assert score.n_total == len(data)
    assert score.status == "complete", (name, score)
    assert score.n_scored == score.n_eligible == len(data)
    assert score.n_missing == score.n_failed == 0
    assert score.coverage == 1
    if name == "confusion":
        assert score.value is None
        assert score.details == {"labels": ["A", "B"], "matrix": [[1, 1], [0, 1]]}
    else:
        assert score.value == pytest.approx(GOLD[name]), (name, score)
    assert score.unit == metric.unit
    assert score.direction == metric.direction
