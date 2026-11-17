"""Deterministic invariants for scores, alignment, and model roundtrips."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from omni_eval import compare, evaluate
from omni_eval.core.align import Aligner
from omni_eval.core.registry import Registry
from omni_eval.types.batch import Batch
from omni_eval.types.case import Case
from omni_eval.types.prob import Prob


@settings(max_examples=25, derandomize=True)
@given(
    st.lists(
        st.tuples(st.sampled_from(["A", "B", "C"]), st.sampled_from(["A", "B", "C"])),
        min_size=1,
        max_size=25,
    )
)
def test_classification_order_and_roundtrip(pairs):
    cases = [Case(id=str(i), expected=t, actual=p) for i, (t, p) in enumerate(pairs)]
    metrics = ["accuracy", "macro_f1", "micro_f1", "weighted_f1", "balanced_accuracy"]
    a = evaluate(cases, task="classification", metrics=metrics)
    b = evaluate(list(reversed(cases)), task="classification", metrics=metrics)
    assert a.summary["metrics"] == b.summary["metrics"]
    assert all(0 <= score.value <= 1 for score in a.scores)
    assert Case.model_validate_json(cases[0].model_dump_json()) == cases[0]
    perfect = [c.model_copy(update={"actual": c.expected}) for c in cases]
    assert all(
        s.value == 1 for s in evaluate(perfect, task="classification", metrics=metrics).scores
    )
    assert compare(a, b, seed=42).intervals == compare(a, b, seed=42).intervals


@settings(max_examples=20, derandomize=True)
@given(st.permutations(["a", "b", "c", "d"]))
def test_rank_monotonicity_and_alignment(order):
    case = Case(
        id="q",
        expected={"relevant_ids": ["a", "c"]},
        contexts=[{"id": item, "rank": i + 1} for i, item in enumerate(order)],
    )
    batch = Batch(task="retrieval", cases=[case])
    for prefix in ("recall", "hit"):
        values = [Registry().get(f"{prefix}@{k}").compute(batch).value for k in range(1, 6)]
        assert values == sorted(values)
    originals = [Case(id=item, expected=item) for item in "abcd"]
    joined = Aligner().join(originals, [{"id": item, "actual": item} for item in order])
    assert [c.actual for c in joined] == list("abcd")


@settings(max_examples=20, derandomize=True)
@given(st.floats(min_value=0, max_value=1, allow_nan=False, allow_infinity=False))
def test_probability_mass_and_coverage(p):
    prob = Prob(labels=["A", "B"], values=[p, 1 - p])
    assert sum(prob.values) == pytest.approx(1)
    with pytest.raises(ValueError):
        Prob(labels=["A", "B"], values=[p + 1, 1 - p])
    cases = [Case(id="a", actual="A", expected="A"), Case(id="b", actual="B", expected="B")]
    full = evaluate(cases, task="classification").metric("accuracy")
    cases[1].actual = None
    partial = evaluate(cases, task="classification").metric("accuracy")
    assert partial.coverage < full.coverage
    assert partial.n_missing == 1
