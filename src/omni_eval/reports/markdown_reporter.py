from omni_eval.schemas.run_result import RunResult


class MarkdownReporter:
    def render(self, run: RunResult) -> str:
        marker = (
            "**DRY RUN / SYNTHETIC — NOT A BENCHMARK**\n\n"
            if run.manifest.is_fixture
            else ""
        )
        lines = [
            f"# OmniEval — {run.manifest.name}",
            "",
            marker,
            "| Metric | Value | Status | Coverage |",
            "|---|---:|---|---:|",
        ]

        for metric in run.metrics:
            lines.append(
                f"| {metric.name} | {metric.value if metric.value is not None else 'N/A'} | {metric.status} | {metric.n_scored}/{metric.n_total} |"
            )

        return "\n".join(lines) + "\n"
