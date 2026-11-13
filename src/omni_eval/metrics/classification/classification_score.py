"""Batch classification, calibration and ranking metrics use corpus-level aggregates."""

from typing import Any

import numpy as np
import sklearn.metrics as sk

from omni_eval.metrics.classification.accuracy import accuracy
from omni_eval.metrics.classification.brier_score import brier_score
from omni_eval.metrics.classification.confusion_matrix import confusion_details
from omni_eval.metrics.classification.expected_calibration_error import (
    calibration_error,
)
from omni_eval.metrics.classification.f1 import classification_f1
from omni_eval.schemas.eval_sample import EvalSample
from omni_eval.schemas.prediction_record import PredictionRecord


def calculate_classification(
    name: str,
    pairs: list[tuple[EvalSample, PredictionRecord]],
    settings: dict[str, Any],
) -> tuple[float | None, dict[str, Any]]:
    y = [str(s.reference["label"]) for s, _ in pairs if s.reference is not None]
    p = [str(pred.output["label"]) for _, pred in pairs]

    details: dict[str, Any] = {}

    if name == "accuracy":
        return accuracy(y, p), details

    if name == "confusion_matrix":
        return None, confusion_details(y, p)

    if name in ("macro_f1", "micro_f1", "weighted_f1"):
        return classification_f1(y, p, name.removesuffix("_f1")), details

    if name in (
        "macro_precision",
        "macro_recall",
        "weighted_precision",
        "weighted_recall",
    ):
        average, operation = name.split("_", 1)
        function = sk.precision_score if operation == "precision" else sk.recall_score

        return float(function(y, p, average=average, zero_division=0)), details

    if name == "balanced_accuracy":
        return float(sk.balanced_accuracy_score(y, p)), details

    if name == "mcc":
        return float(sk.matthews_corrcoef(y, p)), details

    if name == "cohen_kappa":
        return float(sk.cohen_kappa_score(y, p)), details

    if name == "specificity":
        cls = str(settings.get("negative_class", ""))

        if cls not in set(y) or len(set(y) | set(p)) != 2:
            raise ValueError("specificity needs binary labels and negative_class")

        negative = sum(a == cls for a in y)

        return (
            sum(a == cls and b == cls for a, b in zip(y, p, strict=True)) / negative,
            details,
        )

    vectors = [pred.probabilities for _, pred in pairs]

    if any(v is None for v in vectors):
        raise ValueError("probabilities required")

    classes = sorted(vectors[0].labels)  # type: ignore[union-attr]

    if any(
        set(vector.labels) != set(classes) or vector.kind != "softmax"
        for vector in vectors
        if vector is not None
    ):
        raise ValueError(
            "all class distributions must have identical class sets and softmax kind"
        )

    if not set(y).issubset(classes):
        raise ValueError("reference labels absent from class order")

    matrix = [
        [vector.values[vector.labels.index(cls)] for cls in classes]
        for vector in vectors
        if vector is not None
    ]

    if name == "brier_score":
        details["definition"] = (
            "multiclass_sum_squared_error"
            if len(classes) != 2 or not settings.get("positive_class")
            else "binary_positive_class_squared_error"
        )

        return (
            brier_score(
                y,
                classes,
                matrix,
                positive_class=settings.get("positive_class"),
            ),
            details,
        )

    if name in ("ece", "mce"):
        bins = int(settings.get("bins", 10))
        details["bins"] = bins

        return (
            calibration_error(
                y,
                classes,
                matrix,
                bins=bins,
                maximum=name == "mce",
            ),
            details,
        )

    if name == "top_k_accuracy":
        k = int(settings.get("k", 2))

        if not (1 <= k <= len(classes)):
            raise ValueError("top_k_accuracy needs 1 <= k <= number of classes")

        return (
            float(sk.top_k_accuracy_score(y, np.array(matrix), k=k, labels=classes)),
            {"k": k},
        )

    if name == "log_loss":
        if len(classes) < 2:
            raise ValueError("log loss requires two or more classes")

        return float(sk.log_loss(y, matrix, labels=classes)), details

    if name in ("auroc", "average_precision"):
        positive = settings.get("positive_class")

        if len(classes) != 2 or positive not in classes or len(set(y)) < 2:
            raise ValueError(
                "binary AUROC/AP requires positive_class and both reference classes"
            )

        binary = [int(label == positive) for label in y]
        scores = [row[classes.index(positive)] for row in matrix]

        function = sk.roc_auc_score if name == "auroc" else sk.average_precision_score

        return float(function(binary, scores)), {"positive_class": positive}

    raise ValueError(f"unknown classification metric {name}")
