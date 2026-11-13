import pytest
from pydantic import ValidationError

from omni_eval.schemas.agent_trace import AgentTrace
from omni_eval.schemas.asset_ref import AssetRef
from omni_eval.schemas.eval_sample import EvalSample
from omni_eval.schemas.prediction_record import PredictionRecord
from omni_eval.schemas.probability_vector import ProbabilityVector
from omni_eval.schemas.run_spec import RunSpec


@pytest.mark.parametrize("values", [[0.2, 0.2], [float("nan"), 0.1], [-0.1, 1.1]])
def test_invalid_softmax(values):
    with pytest.raises(ValidationError):
        ProbabilityVector(labels=["a", "b"], values=values)


def test_independent_multilabel_does_not_require_simplex():
    vector = ProbabilityVector(labels=["a", "b"], values=[0.8, 0.8], kind="independent")

    assert sum(vector.values) == 1.6


def test_probability_class_order_preserved():
    assert ProbabilityVector(labels=["b", "a"], values=[0.7, 0.3]).labels == ["b", "a"]


def test_unknown_top_level_rejected():
    with pytest.raises(ValidationError):
        EvalSample.model_validate({"sample_id": "x", "input": {}, "arbitrary": 1})


def test_duplicate_agent_index_rejected():
    with pytest.raises(ValidationError):
        AgentTrace.model_validate(
            {
                "steps": [
                    {"index": 0, "kind": "tool_call"},
                    {"index": 0, "kind": "tool_call"},
                ]
            }
        )


def test_context_rank_collision_rejected():
    with pytest.raises(ValidationError):
        PredictionRecord.model_validate(
            {
                "sample_id": "x",
                "output": {},
                "contexts": [
                    {"rank": 1, "chunk_id": "a", "document_id": "d"},
                    {"rank": 1, "chunk_id": "b", "document_id": "d"},
                ],
            }
        )


@pytest.mark.parametrize(
    "uri",
    ["file:///etc/passwd", "ftp://bad/a", "../root/secret", "https://server/../etc"],
)
def test_bad_asset_uri(uri):
    with pytest.raises(ValidationError):
        AssetRef(uri=uri, media_type="text/plain")


def test_bad_runspec_version():
    with pytest.raises(ValidationError):
        RunSpec.model_validate(
            {
                "schema_version": "2.0",
                "source": {"mode": "offline", "predictions": "x"},
                "metrics": [{"name": "accuracy"}],
            }
        )


def test_roundtrip_probability():
    vector = ProbabilityVector(labels=["billing", "support"], values=[0.2, 0.8])

    assert ProbabilityVector.model_validate_json(vector.model_dump_json()) == vector
