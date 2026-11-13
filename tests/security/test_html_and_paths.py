import pytest

from omni_eval.api import evaluate
from omni_eval.exceptions import InputError
from omni_eval.reports.html_reporter import HtmlReporter
from omni_eval.schemas.eval_sample import EvalSample
from omni_eval.schemas.prediction_record import PredictionRecord
from omni_eval.sources.field_mapper import get_path


def test_html_report_escapes_markup():
    sample = EvalSample(sample_id="x", input={"text": "<script>alert(1)</script>"})
    pred = PredictionRecord(sample_id="x", output={"text": "{}"})
    run = evaluate([sample], [pred], ["json_validity"])
    run.manifest.name = '<img src=x onerror="alert(1)">'
    html = HtmlReporter().render(run)

    assert "&lt;img" in html and "<img src=x onerror" not in html


@pytest.mark.parametrize(
    "path",
    ["$.input.__class__()", "__import__('os')", "$.input.x[0]", "$.input.;delete"],
)
def test_field_mapping_rejects_executable_expressions(path):
    with pytest.raises(InputError):
        get_path({"input": {"x": [1]}}, path)
