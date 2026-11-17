"""Per-slice evaluations share the native metric code and suppress small groups."""

import pytest

from omni_eval.core.slices import Slicer
from omni_eval.types.batch import Batch
from omni_eval.types.case import Case
from omni_eval.types.config import Config


def _cases() -> list[Case]:
    return [
        Case(id="a", actual="x", expected="x", meta={"language": "en"}),
        Case(id="b", actual="x", expected="x", meta={"language": "en"}),
        Case(id="c", actual="y", expected="x", meta={"language": "fr"}),
        Case(id="d", actual="y", expected="x", meta={"language": "fr"}),
        Case(id="e", actual="x", expected="x", meta={"language": "fr"}),
        Case(id="f", actual="y", expected="x", meta={"language": "fr"}),
        Case(id="g", actual="x", expected="x"),
    ]


def test_privacy_support_default_suppresses_all_small_groups() -> None:
    grouped = Slicer().group(Batch(task="text", cases=_cases()), "language")
    assert grouped == {}


def test_slice_support_and_no_implicit_missing_bucket() -> None:
    grouped = Slicer().group(Batch(task="text", cases=_cases()), "language", min_count=2)
    assert set(grouped) == {"en", "fr"}
    assert len(grouped["en"].cases) == 2
    assert len(grouped["fr"].cases) == 4


def test_slice_evaluations_have_real_correct_aggregate_scores() -> None:
    cfg = Config(
        task="text",
        data="in-memory",
        metrics=["exact"],
        slices={"keys": ["language"], "min_count": 2},
    )
    from omni_eval.core.engine import Engine

    run = Engine().run(_cases(), cfg)
    score = run.metric("exact")
    assert score.value == pytest.approx(4 / 7)
    groups = score.details["slices"]["language"]
    assert groups["en"] == {"value": 1.0, "status": "complete", "coverage": 1.0, "n_total": 2}
    assert groups["fr"]["value"] == pytest.approx(1 / 4)
    assert groups["fr"]["n_total"] == 4
    assert "unknown" not in groups


def test_slice_defaults_do_not_expose_undersized_groups() -> None:
    cfg = Config(task="text", data="in-memory", metrics=["exact"], slices={"keys": ["language"]})
    from omni_eval.core.engine import Engine

    assert Engine().run(_cases(), cfg).metric("exact").details["slices"] == {"language": {}}


@pytest.mark.parametrize("bad", [0, 1, -1])
def test_invalid_slice_minimum(bad: int) -> None:
    with pytest.raises(ValueError, match="at least two"):
        Slicer().group(Batch(task="text", cases=_cases()), "language", min_count=bad)


def test_invalid_slice_configuration_rejected() -> None:
    from omni_eval.core.engine import Engine

    cfg = Config(task="text", data="in-memory", metrics=["exact"], slices={"keys": "language"})
    with pytest.raises(ValueError, match="list"):
        Engine().run(_cases(), cfg)
