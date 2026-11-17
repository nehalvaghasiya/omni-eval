"""A newcomer may use a class-to-probability dictionary without manual class ordering."""

import pytest

from omni_eval import evaluate
from omni_eval.types.case import Case
from omni_eval.types.prob import Prob


def test_class_probability_mapping_sorted_by_label() -> None:
    case = Case(
        id="a", expected="account", actual="account", probs={"billing": 0.1, "account": 0.9}
    )
    assert case.probs is not None
    assert case.probs.labels == ["account", "billing"]
    assert case.probs.values == [0.9, 0.1]
    score = evaluate([case], task="classification", metrics=["brier", "accuracy"])
    assert score.metric("brier").value == pytest.approx(0.02)
    assert score.metric("accuracy").value == 1


def test_reversed_mapping_order_is_equivalent() -> None:
    a = Case(id="a", probs={"B": 0.7, "A": 0.3})
    b = Case(id="b", probs={"A": 0.3, "B": 0.7})
    assert a.probs == b.probs


def test_ambiguous_probability_form_is_rejected() -> None:
    with pytest.raises(ValueError):
        Case(id="a", probs={"A": 0.8, "B": 0.8})


def test_labeled_probability_model_unchanged() -> None:
    obj = Prob(labels=["B", "A"], values=[0.6, 0.4])
    assert obj.labels == ["B", "A"]
