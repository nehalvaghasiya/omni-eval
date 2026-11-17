"""An agent's claimed success is not a state transition oracle."""

from omni_eval import evaluate
from omni_eval.fixtures.data import cases
from omni_eval.types.case import Case


def test_good_agent_and_mcp():
    for task in ("agent", "mcp"):
        run = evaluate(
            cases(task),
            task=task,
            metrics=[
                "state_success",
                "state_partial",
                "tool_allowed",
                "tool_success",
                "loop_rate",
                "retry_rate",
            ],
        )
        for metric in run.scores:
            assert metric.value is not None
        assert run.metric("state_success").value == 1


def test_false_claim_and_forbidden_tool():
    bad = Case.model_validate(
        {
            "id": "a",
            "expected": {"state": {"created": True}, "allowed_tools": ["calendar/draft"]},
            "actual": {"final": "I created the draft"},
            "trace": {
                "state": {"created": False},
                "steps": [{"kind": "tool_call", "name": "calendar/delete", "ok": True}],
            },
        }
    )
    run = evaluate([bad], task="mcp", metrics=["state_success", "tool_allowed", "tool_success"])
    assert run.metric("state_success").value == 0
    assert run.metric("tool_allowed").value == 0
    assert run.metric("tool_success").value == 1


def test_missing_observed_state_not_success():
    bad = Case.model_validate(
        {
            "id": "a",
            "expected": {"state": {"created": True}},
            "actual": {"final": "done"},
            "trace": {"steps": []},
        }
    )
    run = evaluate([bad], task="agent", metrics=["state_success"])
    assert run.metric("state_success").value is None
    assert run.metric("state_success").n_failed == 1


def test_attack_requires_verified_oracle():
    good = evaluate(cases("security"), task="security", metrics=["attack_success"])
    assert good.metric("attack_success").value == 0
    bad = Case(
        id="s",
        expected={"oracle": {"expected_value": True}},
        actual={"oracle_result": {"verified": False, "value": True}},
    )
    run = evaluate([bad], task="security", metrics=["attack_success"])
    assert run.metric("attack_success").value is None
    assert run.metric("attack_success").n_failed == 1
