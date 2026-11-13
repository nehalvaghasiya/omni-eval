import html

from omni_eval.schemas.run_result import RunResult


class HtmlReporter:
    """HTML-escaped static report; no untrusted markup executes."""

    def render(self, run: RunResult) -> str:
        esc = lambda value: html.escape(str(value), quote=True)
        marker = (
            "<h2>DRY RUN / SYNTHETIC — NOT A BENCHMARK</h2>"
            if run.manifest.is_fixture
            else ""
        )
        rows = "".join(
            f"<tr><td>{esc(m.name)}</td><td>{esc(m.value if m.value is not None else 'N/A')}</td>"
            f"<td>{esc(m.status)}</td><td>{m.n_scored}/{m.n_total}</td></tr>"
            for m in run.metrics
        )
        errors = "".join(f"<li>{esc(error)}</li>" for error in run.errors)

        return (
            '<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            "<title>OmniEval Report</title><style>body{font:16px system-ui;margin:3rem auto;max-width:850px;padding:0 1rem}"
            "table{border-collapse:collapse;width:100%}td,th{padding:.6rem;border-bottom:1px solid #aaa;text-align:left}"
            "th{background:#eee}h2{color:#963}.small{color:#555}</style></head><body>"
            f"<h1>OmniEval: {esc(run.manifest.name)}</h1>{marker}"
            f'<p class="small">Run {esc(run.manifest.run_id)} · {esc(run.manifest.status)} · dataset {esc(run.manifest.dataset_hash)}</p>'
            "<table><thead><tr><th>Metric</th><th>Value</th><th>Status</th><th>Coverage</th></tr></thead>"
            f"<tbody>{rows}</tbody></table><h2>Errors</h2><ul>{errors}</ul></body></html>"
        )
