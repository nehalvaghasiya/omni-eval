"""Refuse unsafe network destinations, artifact injection and HTML scripts."""

import pytest

from omni_eval import evaluate
from omni_eval.reports.html import Html
from omni_eval.sources.policy import HostPolicy
from omni_eval.storage.store import Store
from omni_eval.types.case import Case


def test_ssrf_refuses_metadata_ip():
    with pytest.raises(PermissionError):
        HostPolicy().check_url("http://169.254.169.254/latest/meta-data", ["169.254.169.254"])


def test_ssrf_rejects_unallowed_host():
    with pytest.raises(PermissionError):
        HostPolicy().check_url("https://example.com", ["api.other.test"])


def test_ssrf_rejects_arbitrary_port():
    with pytest.raises(PermissionError):
        HostPolicy().check_url("https://127.0.0.1:8888", ["127.0.0.1"], local_http=True)


def test_script_escaped():
    run = evaluate(
        [Case(id="x", input="<script>alert(1)</script>", actual="ok", expected="ok")], task="text"
    )
    assert "<script>" not in Html().render(run)


def test_symlinked_artifact_rejected(tmp_path):
    run = evaluate([Case(id="a", expected="x", actual="x")], task="text", out=tmp_path)
    file = run.path / "report.html"
    file.unlink()
    file.symlink_to(tmp_path / "outside")
    with pytest.raises(ValueError):
        Store().verify(run.path)
