"""Paired comparison, point identity, confidence and CI gates."""

import pytest

from omni_eval import compare, evaluate
from omni_eval.core.gates import GateEngine
from omni_eval.fixtures.data import cases
from omni_eval.types.case import Case
from omni_eval.types.gate import GateRule


def test_deterministic_bootstrap():
    base = cases("classification")
    left = evaluate(base, task="classification", metrics=["accuracy", "macro_f1"])
    updated = [c.model_copy(update={"actual": {"label": "A"}}) if c.id == "c2" else c for c in base]
    right = evaluate(
        list(reversed(updated)), task="classification", metrics=["accuracy", "macro_f1"]
    )
    first = compare(left, right, seed=5)
    second = compare(left, right, seed=5)
    assert first.intervals == second.intervals
    assert first.deltas["accuracy"] == pytest.approx(1 / 3)
    assert first.paired_n["accuracy"] == 3
    assert first.paired_n["macro_f1"] == 0


def test_changed_ground_truth_fails():
    a = evaluate([Case(id="a", expected="a", actual="a")], task="text")
    b = evaluate([Case(id="a", expected="b", actual="a")], task="text")
    with pytest.raises(ValueError, match="different reference"):
        compare(a, b)


def test_thresholds():
    run = evaluate(cases("classification"), task="classification")
    rules = [
        GateRule(metric="accuracy", value=0.5),
        GateRule(metric="macro_f1", value=0.9),
        GateRule(metric="missing", value=0.1),
    ]
    results = GateEngine().check(run, rules)
    assert [x.status for x in results] == ["pass", "fail", "inconclusive"]
