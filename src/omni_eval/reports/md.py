"""Minimal text/Markdown score summary."""

from omni_eval.types.run import Run


class Markdown:
    def render(self, run: Run) -> str:
        prefix = "DEMO / SYNTHETIC - " if run.fixture else ""
        rows = "".join(
            f"| {s.name} | {(s.value if s.value is not None else 'N/A')} | {s.status} | {s.coverage:.1%} |\n"
            for s in run.scores
        )
        return (
            f"# {prefix}OmniEval\n\n| Metric | Value | Status | Coverage |\n|---|---:|---|---:|\n"
            + rows
        )
