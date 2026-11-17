"""Never present non-paired corpus deltas as fair model comparisons."""

from omni_eval import compare, evaluate
from omni_eval.types.case import Case


def test_mismatched_case_ids_withhold_corpus_delta() -> None:
    left = evaluate(
        [Case(id="a", expected="x", actual="x"), Case(id="b", expected="x", actual="x")],
        task="text",
    )
    right = evaluate([Case(id="a", expected="x", actual="y")], task="text")
    diff = compare(left, right)
    assert diff.paired_n["exact"] == 1
    assert diff.deltas["exact"] == -1.0
    assert diff.deltas["token_f1"] == -1.0
    assert any("paired subset" in message for message in diff.warnings)


def test_corpus_delta_withheld_for_mismatched_dataset() -> None:
    left = evaluate(
        [
            Case(id="a", expected={"label": "x"}, actual={"label": "x"}),
            Case(id="b", expected={"label": "y"}, actual={"label": "y"}),
        ],
        task="classification",
    )
    right = evaluate(
        [Case(id="a", expected={"label": "x"}, actual={"label": "x"})],
        task="classification",
    )
    diff = compare(left, right)
    assert diff.deltas["macro_f1"] is None
    assert any("withheld" in message for message in diff.warnings)


def test_missing_predicted_row_annotates_unequal_coverage() -> None:
    left = evaluate(
        [Case(id="a", expected="x", actual="x"), Case(id="b", expected="x", actual="x")],
        task="text",
    )
    right = evaluate(
        [Case(id="a", expected="x", actual="x"), Case(id="b", expected="x")],
        task="text",
    )
    diff = compare(left, right)
    assert any("coverage" in message for message in diff.warnings)
