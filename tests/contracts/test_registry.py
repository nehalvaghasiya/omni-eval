"""Every declared native metric is actually resolvable and named safely."""

import pytest

from omni_eval.core.planner import Planner
from omni_eval.core.registry import Registry
from omni_eval.fixtures.data import cases
from omni_eval.types.config import Config


def test_registry_only_contains_implemented_metrics():
    reg = Registry()
    for name in reg.list():
        metric = reg.get(name)
        assert metric.name == name
        assert metric.tasks
        assert reg.describe(name)["version"]


@pytest.mark.parametrize(
    "task", ["classification", "text", "retrieval", "rag", "agent", "mcp", "audio", "vision"]
)
def test_task_defaults(task):
    names = Planner().build(Config(task=task, data="synthetic"), cases(task))
    assert names and all(Registry().get(name) for name in names)


def test_reference_policy_before_endpoint_calls():
    from omni_eval.types.case import Case

    with pytest.raises(ValueError, match="required references"):
        Planner().build(
            Config(task="classification", data="synthetic", metrics=["accuracy"]),
            [Case(id="x", actual="A")],
        )


def test_readme_documents_every_registered_metric():
    from pathlib import Path

    readme = (Path(__file__).resolve().parents[2] / "README.md").read_text(encoding="utf-8")
    assert [name for name in Registry().list() if f"`{name}`" not in readme] == []
