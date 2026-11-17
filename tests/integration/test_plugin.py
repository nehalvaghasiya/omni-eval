"""Fixture subprocess checks transport, not upstream framework quality."""

import pytest

from omni_eval import evaluate
from omni_eval.fixtures.data import cases
from omni_eval.plugins.catalog import Catalog
from omni_eval.plugins.reply import WorkerReply
from omni_eval.types.config import Config


def test_real_fixture_process():
    conf = Config(task="text", data="memory", metrics=["fixture:echo"])
    from omni_eval.core.engine import Engine

    run = Engine().run(cases("text"), conf, fixture=True)
    assert run.metric("fixture:echo").value == 1
    assert run.metric("fixture:echo").source == "fixture"


def test_planned_provider_is_refused_before_scoring():
    with pytest.raises(RuntimeError, match="not prepared"):
        evaluate(cases("rag"), task="rag", metrics=["ragas:faithfulness"])


def test_protocol_unknown_version():
    with pytest.raises(ValueError):
        WorkerReply(protocol=5, job="x", provider="fixture", status="complete", metrics=[])


def test_no_fictitious_support():
    assert Catalog().get("ragas")["qualified"] is False
    assert Catalog().get("fixture")["qualified"] is False


def test_suite_refused_before_file_access():
    conf = Config(task="security", plugin="promptfoo", suite="synthetic-suite")
    with pytest.raises(RuntimeError, match="not prepared"):
        evaluate(conf)


def test_suite_dry_run_uses_only_synthetic_worker():
    conf = Config(task="security", plugin="promptfoo", suite="synthetic-suite")
    result = evaluate(conf, dry_run=True)
    assert result.fixture is True
    assert result.metric("fixture:echo").source == "fixture"


def test_fixture_metric_always_marks_the_run_synthetic():
    run = evaluate(cases("text"), task="text", metrics=["fixture:echo"])
    assert run.fixture
    assert run.manifest["synthetic"]
