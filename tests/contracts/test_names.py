"""One class per source file and short source names, with ecosystem exceptions."""

import ast
from pathlib import Path


def test_source_classes_and_names():
    base = Path(__file__).resolve().parents[2] / "src/omni_eval"
    exceptions = {"__init__", "__main__"}
    for file in base.rglob("*.py"):
        if file.stem not in exceptions:
            parts = file.stem.split("_")
            assert 1 <= len(parts) <= 2 and all(
                part.islower() and part.isidentifier() for part in parts
            ), file
        tree = ast.parse(file.read_text(encoding="utf-8"))
        classes = [
            node
            for node in tree.body
            if isinstance(node, (ast.ClassDef, ast.AsyncFunctionDef))
            and isinstance(node, ast.ClassDef)
        ]
        assert len(classes) <= 1, file
