"""All report formats are accessible, valid, and do not execute untrusted text."""

import xml.etree.ElementTree as ET

from omni_eval import evaluate
from omni_eval.reports.html import Html
from omni_eval.reports.json import Json
from omni_eval.reports.junit import Junit
from omni_eval.reports.md import Markdown
from omni_eval.types.case import Case


def test_reports_preserve_real_scores_and_sources() -> None:
    run = evaluate([Case(id="x", expected="ok", actual="ok")], task="text")
    data = Json().render(run)
    assert data["scores"][0]["value"] == 1
    assert data["scores"][0]["source"] == "native"
    assert "exact" in Markdown().render(run)
    suite = ET.fromstring(Junit().render(run))
    assert suite.attrib["tests"] == "2"
    assert len(suite.findall("testcase")) == 2


def test_html_escapes_unsafe_metric_names() -> None:
    run = evaluate([Case(id="x", expected="a", actual="a")], task="text")
    run.scores[0].name = "<img src=x onerror=alert(1)>"
    page = Html().render(run)
    assert "<img src=x onerror=alert(1)>" not in page
    assert "&lt;img" in page


def test_synthetic_marking_in_every_report() -> None:
    run = evaluate("ignored.jsonl", task="text", dry_run=True)
    assert run.fixture
    assert "SYNTHETIC" in Html().render(run)
    assert "SYNTHETIC" in Markdown().render(run)
    assert Json().render(run)["fixture"] is True
    assert "SYNTHETIC" in Junit().render(run)


def test_junit_does_not_report_scorer_errors_as_success():
    run = evaluate([Case(id="bad", actual="not a transcript", expected="")], task="audio")
    suite = ET.fromstring(Junit().render(run))
    assert len(suite.findall("testcase/error")) == 2
