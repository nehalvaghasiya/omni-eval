from omni_eval.schemas.agent_trace import AgentTrace


def trajectory_check(trace: AgentTrace, expected: list[str]) -> float:
    """The required ordered tool subsequence must appear in the trace."""
    observed = [s.tool for s in trace.steps if s.kind == "tool_call"]
    iterator = iter(observed)

    return float(
        all(any(tool == required for tool in iterator) for required in expected)
    )
