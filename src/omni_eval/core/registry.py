"""Explicit metric registry; every name resolves to its owning class."""

import re
from typing import Any

from omni_eval.metrics.agent.allowed import ToolAllowed
from omni_eval.metrics.agent.loop import LoopRate
from omni_eval.metrics.agent.order import StepOrder
from omni_eval.metrics.agent.partial import StatePartial
from omni_eval.metrics.agent.retry import RetryRate
from omni_eval.metrics.agent.state import StateSuccess
from omni_eval.metrics.agent.success import ToolSuccess
from omni_eval.metrics.base import Metric
from omni_eval.metrics.classify.accuracy import Accuracy
from omni_eval.metrics.classify.ap import Ap
from omni_eval.metrics.classify.auc import Auc
from omni_eval.metrics.classify.balanced import Balanced
from omni_eval.metrics.classify.brier import Brier
from omni_eval.metrics.classify.confusion import Confusion
from omni_eval.metrics.classify.ece import Ece
from omni_eval.metrics.classify.f1 import F1
from omni_eval.metrics.classify.kappa import Kappa
from omni_eval.metrics.classify.logloss import LogLoss
from omni_eval.metrics.classify.mcc import Mcc
from omni_eval.metrics.classify.mce import Mce
from omni_eval.metrics.classify.precision import Precision
from omni_eval.metrics.classify.recall import Recall
from omni_eval.metrics.classify.topk import TopK
from omni_eval.metrics.code.syntax import Syntax
from omni_eval.metrics.perf.latency import Latency
from omni_eval.metrics.perf.reliability import Reliability
from omni_eval.metrics.rag.citation import CitationValid
from omni_eval.metrics.rag.context import ContextCoverage
from omni_eval.metrics.rag.evidence import Evidence
from omni_eval.metrics.rag.rerank import RerankDelta
from omni_eval.metrics.retrieval.hit import HitAtK
from omni_eval.metrics.retrieval.map import Map
from omni_eval.metrics.retrieval.mrr import Mrr
from omni_eval.metrics.retrieval.ndcg import Ndcg
from omni_eval.metrics.retrieval.precision import PrecisionAtK
from omni_eval.metrics.retrieval.recall import RecallAtK
from omni_eval.metrics.retrieval.rprec import RPrec
from omni_eval.metrics.safety.attack import AttackSuccess
from omni_eval.metrics.safety.policy import PolicyViolation
from omni_eval.metrics.sql.parse import Parse
from omni_eval.metrics.text.cer import Cer
from omni_eval.metrics.text.exact import Exact
from omni_eval.metrics.text.fields import Fields
from omni_eval.metrics.text.format import Format
from omni_eval.metrics.text.json import JsonValid
from omni_eval.metrics.text.loss import Loss
from omni_eval.metrics.text.schema import SchemaValid
from omni_eval.metrics.text.token_f1 import TokenF1
from omni_eval.metrics.text.wer import Wer

_CLASSES: dict[str, type[Metric]] = {
    "accuracy": Accuracy,
    **{name: F1 for name in ("macro_f1", "micro_f1", "weighted_f1")},
    **{
        name: Precision
        for name in ("precision", "macro_precision", "micro_precision", "weighted_precision")
    },
    **{name: Recall for name in ("recall", "macro_recall", "micro_recall", "weighted_recall")},
    "balanced_accuracy": Balanced,
    "mcc": Mcc,
    "kappa": Kappa,
    "confusion": Confusion,
    "brier": Brier,
    "logloss": LogLoss,
    "ece": Ece,
    "mce": Mce,
    "topk@k": TopK,
    "auroc": Auc,
    "auprc": Ap,
    "exact": Exact,
    "answer_match": Exact,
    "token_f1": TokenF1,
    "token_precision": TokenF1,
    "token_recall": TokenF1,
    "json_valid": JsonValid,
    "schema_valid": SchemaValid,
    "fields_f1": Fields,
    "wer": Wer,
    "cer": Cer,
    "format": Format,
    "nll": Loss,
    "perplexity": Loss,
    "code_valid": Syntax,
    "sql_select_safe": Parse,
    "precision@k": PrecisionAtK,
    "recall@k": RecallAtK,
    "hit@k": HitAtK,
    "success@k": HitAtK,
    "mrr": Mrr,
    "map": Map,
    "ndcg@k": Ndcg,
    "rprec": RPrec,
    "citation_valid": CitationValid,
    "evidence_coverage": Evidence,
    "context_precision": ContextCoverage,
    "rerank_delta": RerankDelta,
    "state_success": StateSuccess,
    "state_partial": StatePartial,
    "tool_allowed": ToolAllowed,
    "tool_success": ToolSuccess,
    "loop_rate": LoopRate,
    "step_order": StepOrder,
    "retry_rate": RetryRate,
    "attack_success": AttackSuccess,
    "policy_violation": PolicyViolation,
    "latency_mean": Latency,
    "latency_p50": Latency,
    "latency_p95": Latency,
    "latency_p99": Latency,
    "success_rate": Reliability,
}


class Registry:
    def get(self, name: str) -> Metric:
        match = re.fullmatch("(precision|recall|hit|success|ndcg|topk)@(\\d+)", name)
        key = match.group(1) + "@k" if match else name
        k = int(match.group(2)) if match else None
        if k is not None and (not 1 <= k <= 1000):
            raise ValueError("k must be in the range 1..1000")
        klass = _CLASSES.get(key)
        if klass is None:
            if ":" in name:
                raise ValueError(
                    f"external metric {name!r} requires prepared plugin; no native fallback"
                )
            raise ValueError(f"unknown metric {name!r}; run 'omni-eval metrics list'")
        return klass(name=name, k=k)

    def list(self, task: str | None = None) -> list[str]:
        return sorted(
            (name for name, klass in _CLASSES.items() if task is None or task in klass.tasks)
        )

    def describe(self, name: str) -> dict[str, Any]:
        metric = self.get(name)
        return {
            "name": name,
            "tasks": sorted(metric.tasks),
            "needs": sorted(metric.needs),
            "version": metric.version,
            "direction": metric.direction,
            "unit": metric.unit,
            "level": metric.level,
            "description": metric.description or type(metric).__doc__ or "",
        }
