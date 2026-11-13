import pytest

from omni_eval.api import evaluate
from omni_eval.exceptions import InputError
from omni_eval.schemas.eval_sample import EvalSample
from omni_eval.schemas.prediction_record import PredictionRecord


def test_rag_retrieval_and_fake_citation():
    sample = EvalSample(
        sample_id="rag",
        input={"query": "refund"},
        reference={"relevant_chunk_ids": ["policy#4"]},
    )
    prediction = PredictionRecord(
        sample_id="rag",
        output={"answer": "30 days", "citations": ["fake", "policy#4"]},
        contexts=[{"document_id": "policy", "chunk_id": "policy#4", "rank": 1}],
    )
    run = evaluate(
        [sample], [prediction], ["recall_at_k", "ndcg_at_k", "citation_validity"]
    )

    assert [metric.value for metric in run.metrics] == [1, 1, 0.5]


def test_reference_free_retrieval_recall_rejected():
    sample = EvalSample(sample_id="rag", input={"query": "refund"})
    pred = PredictionRecord(sample_id="rag", output={}, contexts=[])

    with pytest.raises(InputError, match="reference.relevant_chunk_ids"):
        evaluate([sample], [pred], ["recall_at_k"])


def test_agent_false_claim_fails():
    s = EvalSample(
        sample_id="agent",
        input={"task": "save"},
        reference={
            "expected_state": {"saved": True},
            "allowed_tools": ["calendar/create_draft"],
            "expected_tools": ["calendar/create_draft"],
        },
    )
    p = PredictionRecord(
        sample_id="agent",
        output={"final": "saved"},
        trace={
            "steps": [
                {
                    "index": 0,
                    "kind": "tool_call",
                    "tool": "calendar/create_draft",
                    "result": {"ok": False},
                },
                {"index": 1, "kind": "tool_call", "tool": "calendar/delete_event"},
            ],
            "observed_state": {"saved": False},
        },
    )
    r = evaluate(
        [s], [p], ["agent_state_success", "unauthorized_tool_rate", "trajectory_check"]
    )

    assert [m.value for m in r.metrics] == [0, 0.5, 1]


def test_no_agent_observed_state_fails_incomplete():
    s = EvalSample(sample_id="a", input={}, reference={"expected_state": {"ok": True}})
    p = PredictionRecord(sample_id="a", output={"final": "ok"}, trace={"steps": []})
    r = evaluate([s], [p], ["agent_state_success"])

    assert r.metrics[0].value is None
    assert r.metrics[0].status == "insufficient_input"


def test_attack_success_requires_verified_oracle():
    s = EvalSample(
        sample_id="attack",
        input={"payload": "synthetic"},
        reference={"success_oracle": {"expected_value": True}},
    )
    p = PredictionRecord(
        sample_id="attack",
        output={"oracle_result": {"detector_flag": True, "value": True}},
    )
    r = evaluate([s], [p], ["attack_success"])

    assert r.metrics[0].value is None
    p.output["oracle_result"]["verified"] = True

    assert evaluate([s], [p], ["attack_success"]).metrics[0].value == 1
