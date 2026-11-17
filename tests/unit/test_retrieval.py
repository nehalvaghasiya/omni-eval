"""Known retrieval qrels, rank ties and denominator conventions."""

import pytest

from omni_eval import evaluate
from omni_eval.fixtures.data import cases
from omni_eval.types.case import Case


def test_retrieval_gold():
    run = evaluate(
        cases("retrieval"),
        task="retrieval",
        metrics=["precision@2", "recall@2", "hit@2", "mrr", "map", "ndcg@2", "rprec"],
    )
    assert run.metric("precision@2").value == 0.5
    assert run.metric("recall@2").value == 0.5
    assert run.metric("hit@2").value == 1
    assert run.metric("mrr").value == 1
    assert run.metric("map").value == 0.5
    assert run.metric("rprec").value == 0.5
    assert 0 < run.metric("ndcg@2").value < 1


def test_bad_ranks():
    with pytest.raises(ValueError, match="unique"):
        Case.model_validate(
            {"id": "x", "contexts": [{"id": "a", "rank": 1}, {"id": "b", "rank": 1}]}
        )


def test_no_qrels_rejected():
    data = cases("retrieval")
    data[0] = data[0].model_copy(update={"expected": None})
    with pytest.raises(ValueError):
        evaluate(data, task="retrieval", metrics=["recall@5"])


def test_no_fabricated_recall_from_context_text():
    case = Case(
        id="x", actual={"answer": "yes"}, contexts=[{"id": "a", "rank": 1, "text": "relevant?"}]
    )
    with pytest.raises(ValueError, match="required references"):
        evaluate([case], task="rag", metrics=["recall@1"])


def test_mrr_and_map_do_not_silently_truncate_at_five():
    case = Case(
        id="long",
        expected={"relevant_ids": ["d6"]},
        contexts=[{"id": f"d{i}", "rank": i} for i in range(1, 7)],
    )
    run = evaluate([case], task="retrieval", metrics=["mrr", "map"])
    assert run.metric("mrr").value == pytest.approx(1 / 6)
    assert run.metric("map").value == pytest.approx(1 / 6)
