"""Task field access shared by independent metric classes."""

from typing import Any

from omni_eval.types.case import Case


def field(value: Any, *keys: str) -> Any:
    if isinstance(value, dict):
        for key in keys:
            if key in value:
                return value[key]
    return value if not isinstance(value, dict) else None


def label(case: Case, *, truth: bool = False) -> Any:
    return field(case.expected if truth else case.actual, "label", "class")


def answer(case: Case, *, truth: bool = False) -> str | None:
    value = field(case.expected if truth else case.actual, "answer", "text", "value", "final")
    return value if isinstance(value, str) else None


def relevant(case: Case) -> set[str] | None:
    if not isinstance(case.expected, dict):
        return None
    ids = case.expected.get("relevant_ids", case.expected.get("relevant_chunk_ids"))
    if ids is None:
        return None
    if not isinstance(ids, list) or len(set(ids)) != len(ids):
        raise ValueError("expected.relevant_ids must contain unique document IDs")
    return set(ids)


def ranks(case: Case) -> list[str]:
    return [x.id for x in sorted(case.contexts, key=lambda x: x.rank)]


def citations(case: Case) -> list[str] | None:
    if isinstance(case.actual, dict):
        got = case.actual.get("citations")
        if isinstance(got, list):
            return [str(x) for x in got]
    return None
