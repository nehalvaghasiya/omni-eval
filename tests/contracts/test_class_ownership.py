"""Keep the original one top-level class per module rule enforceable."""

import ast
import re
from pathlib import Path


def test_no_module_owns_multiple_classes():
    root = Path(__file__).resolve().parents[2] / "src" / "omni_eval"
    offenders = []

    for module in root.rglob("*.py"):
        tree = ast.parse(module.read_text(encoding="utf-8"))
        count = sum(isinstance(node, ast.ClassDef) for node in tree.body)

        if count > 1:
            offenders.append(str(module.relative_to(root)))

    assert not offenders, offenders


def test_class_filename_corresponds_to_class_name():
    root = Path(__file__).resolve().parents[2] / "src" / "omni_eval"

    for module in root.rglob("*.py"):
        for node in ast.parse(module.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.ClassDef):
                snake = re.sub(
                    r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])",
                    "_",
                    node.name.replace("OpenAI", "Openai").replace(
                        "DeepEval", "Deepeval"
                    ),
                ).lower()

                assert module.stem == snake, (
                    f"{module} class {node.name} violates one-class filename rule"
                )
