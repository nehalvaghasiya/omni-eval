"""Accessible escaped HTML, no execution of generated model text."""

from html import escape

from omni_eval.types.run import Run


class Html:
    def render(self, run: Run) -> str:
        title = "DEMO / SYNTHETIC - " if run.fixture else ""
        rows = "".join(
            "<tr><th scope='row'>"
            + escape(s.name)
            + "</th><td>"
            + (format(s.value, ".6g") if s.value is not None else "unavailable")
            + "</td><td>"
            + escape(s.status)
            + "</td><td>"
            + format(s.coverage, ".1%")
            + "</td></tr>"
            for s in run.scores
        )
        return (
            "<!doctype html><html lang='en'><head><meta charset='utf-8'><title>"
            + escape(title + "OmniEval")
            + "</title><style>body{font:16px system-ui;margin:3rem auto;max-width:56rem;padding:0 1rem}table{border-collapse:collapse;width:100%}td,th{border:1px solid #bbb;padding:.7rem;text-align:left}caption{font-weight:700;text-align:left;margin-bottom:1rem}</style></head><body><main>"
            + "<h1>"
            + escape(title + "OmniEval")
            + "</h1><p>"
            + escape(run.task)
            + " · "
            + escape(run.status)
            + "</p><table><caption>Evaluation scores and coverage</caption><thead><tr><th scope='col'>Metric</th><th scope='col'>Value</th><th scope='col'>Status</th><th scope='col'>Coverage</th></tr>"
            + "</thead><tbody>"
            + rows
            + "</tbody></table></main></body></html>"
        )
