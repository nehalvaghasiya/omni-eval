"""A synthetic run validates the user's selected metric rather than silently replacing it."""

import pytest

from omni_eval import evaluate


def test_dry_run_preserves_custom_k() -> None:
    run = evaluate("no-file.jsonl", task="retrieval", metrics=["recall@1"], dry_run=True)
    assert [score.name for score in run.scores] == ["recall@1"]


def test_dry_run_rejects_incompatible_metric() -> None:
    with pytest.raises(ValueError, match="not valid for task"):
        evaluate("no-file.jsonl", task="classification", metrics=["wer"], dry_run=True)


def test_dry_run_rejects_unknown_metric() -> None:
    with pytest.raises(ValueError, match="unknown metric"):
        evaluate("no-file.jsonl", task="text", metrics=["not_real"], dry_run=True)
