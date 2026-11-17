"""CSV nested fields, quoted commas and safe mapping contracts."""

import csv

import pytest

from omni_eval import evaluate
from omni_eval.io.csv import Csv
from omni_eval.types.config import Config


def test_csv_with_commas_and_newlines(tmp_path):
    path = tmp_path / "cases.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["case_id", "input_col", "gold", "prediction"])
        writer.writerow(["x1", "a,b\nsecond line", "hello", "hello"])
        writer.writerow(["x2", "你好", "hi", "hello"])
    conf = Config(
        task="text",
        data=str(path),
        metrics=["exact"],
        columns={"id": "case_id", "input": "input_col", "expected": "gold", "actual": "prediction"},
    )
    result = evaluate(conf)
    assert result.metric("exact").value == 0.5
    assert result.cases[0].input == "a,b\nsecond line"


def test_csv_missing_mapping(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text("id,actual\n1,hi\n")
    with pytest.raises(ValueError, match="mapping"):
        list(Csv().read(p, {"actual": "actual"}))


def test_parquet_requires_optional_dependency(tmp_path, monkeypatch):
    import sys

    from omni_eval.io.parquet import Parquet

    monkeypatch.setitem(sys.modules, "pyarrow", None)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", None)
    with pytest.raises(RuntimeError, match="parquet"):
        list(Parquet().read(tmp_path / "x.parquet"))


def test_parquet_real_roundtrip_matches_jsonl(tmp_path):
    pa = pytest.importorskip("pyarrow", reason="optional parquet extra is absent")
    import pyarrow.parquet as pq

    from omni_eval.io.jsonl import Jsonl

    rows = [
        {"id": "a", "expected": "hello", "actual": "hello"},
        {"id": "b", "expected": "你好", "actual": "world"},
    ]
    parquet = tmp_path / "cases.parquet"
    jsonl = tmp_path / "cases.jsonl"
    pq.write_table(pa.Table.from_pylist(rows), parquet)
    Jsonl().write(jsonl, rows)
    a = evaluate(parquet, task="text")
    b = evaluate(jsonl, task="text")
    assert a.summary["metrics"] == b.summary["metrics"]
    assert a.metric("exact").value == 0.5
