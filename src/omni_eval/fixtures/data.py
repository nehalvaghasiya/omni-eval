"""Small synthetic examples with known oracles; never model benchmark data."""

from omni_eval.types.case import Case

_FIXTURES: dict[str, list[dict[str, object]]] = {
    "classification": [
        {
            "id": "c1",
            "input": "request a",
            "expected": {"label": "A"},
            "actual": {"label": "A"},
            "probs": {"labels": ["A", "B"], "values": [0.9, 0.1]},
        },
        {
            "id": "c2",
            "input": "request b",
            "expected": {"label": "A"},
            "actual": {"label": "B"},
            "probs": {"labels": ["A", "B"], "values": [0.2, 0.8]},
        },
        {
            "id": "c3",
            "input": "request c",
            "expected": {"label": "B"},
            "actual": {"label": "B"},
            "probs": {"labels": ["A", "B"], "values": [0.4, 0.6]},
        },
    ],
    "text": [{"id": "t1", "input": "sample", "expected": "hello world", "actual": "Hello, world!"}],
    "retrieval": [
        {
            "id": "r1",
            "input": "query",
            "expected": {"relevant_ids": ["d1", "d3"]},
            "actual": {"answer": ""},
            "contexts": [
                {"id": "d1", "rank": 1, "text": "answer"},
                {"id": "d2", "rank": 2, "text": "other"},
            ],
        }
    ],
    "rag": [
        {
            "id": "g1",
            "input": "refund?",
            "expected": {"answer": "30 days", "relevant_ids": ["d1"]},
            "actual": {"answer": "30 days", "citations": ["d1"]},
            "contexts": [{"id": "d1", "rank": 1, "text": "30 days"}],
        }
    ],
    "agent": [
        {
            "id": "a1",
            "input": "create draft",
            "expected": {
                "state": {"draft_created": True},
                "allowed_tools": ["calendar.create_draft"],
            },
            "actual": {"final": "done"},
            "trace": {
                "state": {"draft_created": True},
                "steps": [{"kind": "tool_call", "name": "calendar.create_draft", "ok": True}],
            },
        }
    ],
    "mcp": [
        {
            "id": "m1",
            "input": "create draft",
            "expected": {
                "state": {"draft_created": True},
                "allowed_tools": ["calendar/create_draft"],
            },
            "actual": {"final": "done"},
            "trace": {
                "state": {"draft_created": True},
                "steps": [{"kind": "tool_call", "name": "calendar/create_draft", "ok": True}],
            },
        }
    ],
    "audio": [
        {"id": "w1", "input": "transcript", "expected": "hello world", "actual": "hello word"}
    ],
    "vision": [
        {
            "id": "v1",
            "input": {"asset": "example.png"},
            "expected": {"label": "cat"},
            "actual": {"label": "cat"},
        }
    ],
    "security": [
        {
            "id": "s1",
            "input": "simulated prompt injection",
            "expected": {"oracle": {"expected_value": True}},
            "actual": {"oracle_result": {"verified": True, "value": False}},
        }
    ],
    "code": [{"id": "p1", "input": "write Python", "actual": "print('ok')"}],
    "sql": [{"id": "q1", "input": "query db", "actual": "SELECT 1;"}],
}


def cases(task: str) -> list[Case]:
    if task not in _FIXTURES:
        raise ValueError(f"unknown demo task {task!r}; choose {sorted(_FIXTURES)}")
    return [Case.model_validate(row) for row in _FIXTURES[task]]


def names() -> list[str]:
    return sorted(_FIXTURES)
