"""Transform legacy and simple inline records into one Case."""

from typing import Any

from omni_eval.types.case import Case


class Mapper:
    def to_case(self, raw: dict[str, Any]) -> Case:
        if "id" in raw:
            return Case.model_validate(raw)
        if "sample_id" in raw:
            data = {
                "id": raw["sample_id"],
                "input": raw.get("input"),
                "expected": raw.get("reference"),
                "meta": raw.get("metadata", {}),
            }
            return Case.model_validate(data)
        raise ValueError("missing case id (or legacy sample_id)")

    def to_actual(self, raw: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        sid = raw.get("id", raw.get("sample_id"))
        if not isinstance(sid, str) or not sid.strip():
            raise ValueError("prediction lacks valid id or sample_id")
        actual = raw.get("actual", raw.get("output"))
        probs = raw.get("probs", raw.get("probabilities"))
        contexts = raw.get("contexts", [])
        if contexts:
            contexts = [
                {
                    "id": x.get("id", x.get("chunk_id")),
                    "rank": x.get("rank"),
                    "text": x.get("text"),
                    "score": x.get("score"),
                    "source": x.get("source"),
                }
                for x in contexts
            ]
        trace = raw.get("trace")
        if isinstance(trace, dict) and "observed_state" in trace:
            trace = {
                "state": trace.get("observed_state"),
                "steps": [
                    {
                        "kind": x.get("kind", "tool_call"),
                        "name": x.get("name", x.get("tool")),
                        "args": x.get("args", x.get("arguments", {})),
                        "result": x.get("result"),
                        "ok": (x.get("result") or {}).get("ok")
                        if isinstance(x.get("result"), dict)
                        else None,
                    }
                    for x in trace.get("steps", [])
                ],
            }
        timing = raw.get("timing")
        if isinstance(timing, dict):
            timing = {
                "latency_ms": timing.get("latency_ms", 0),
                "source": timing.get("source", "measured_external"),
                "scope": timing.get("scope", "end_to_end"),
            }
        return (
            sid,
            {
                "actual": actual,
                "probs": probs,
                "contexts": contexts or [],
                "trace": trace,
                "timing": timing,
                "error": raw.get("error"),
            },
        )
