"""Rubric-based, opt-in judge metric with per-case evidence."""

from pathlib import Path

import httpx
import yaml

from omni_eval.io.hash import digest
from omni_eval.plugins.broker import Broker
from omni_eval.plugins.budget import Budget
from omni_eval.types.batch import Batch
from omni_eval.types.config import Config
from omni_eval.types.judge import Judge
from omni_eval.types.point import Point
from omni_eval.types.rubric import Rubric
from omni_eval.types.score import Score


class JudgePlugin:
    """Estimate a rubric score, never relabel it as reference-grounded accuracy."""

    name = "judge"
    source = "judge"
    tasks = frozenset({"text", "rag", "agent", "mcp", "security"})

    def validate(self, name: str, config: Config) -> tuple[Judge, Rubric]:
        """Resolve authorization and rubric before making a target model call."""
        if not isinstance(config.judge, dict):
            raise ValueError("judge metric needs an explicit judge configuration")
        judge = Judge.model_validate(config.judge)
        if not judge.approve_external_data:
            raise PermissionError("judge data forwarding needs approve_external_data: true")
        rubric_data = yaml.safe_load(Path(judge.rubric).read_text(encoding="utf-8"))
        rubric = Rubric.model_validate(rubric_data)
        if name != f"judge:{rubric.id}":
            raise ValueError("requested judge metric must match configured rubric ID")
        if config.task not in self.tasks:
            raise ValueError(f"judge metric is not supported for {config.task}")
        return judge, rubric

    def score(self, name: str, batch: Batch, config: Config) -> Score:
        judge, rubric = self.validate(name, config)
        broker = Broker(judge)
        budget = Budget(judge.max_calls, judge.max_tokens)
        points: list[Point] = []
        values: list[float] = []
        failures = 0
        revisions: set[str] = set()
        tokens = 0
        for case in batch.cases:
            missing = [field for field in rubric.required if case.get(field) is None]
            if missing:
                points.append(
                    Point(id=case.id, metric=name, status="missing_input", reason=",".join(missing))
                )
                continue
            try:
                budget.reserve(judge.per_call_tokens)
                untrusted = {field: case.get(field) for field in rubric.required}
                verdict = broker.structured(rubric.criterion, untrusted)
                if verdict.status == "abstain":
                    points.append(
                        Point(id=case.id, metric=name, status="abstain", reason=verdict.reason)
                    )
                    continue
                if verdict.score is None or not rubric.low <= verdict.score <= rubric.high:
                    raise ValueError("judge score outside rubric range")
                if rubric.require_evidence:
                    valid_ids = {context.id for context in case.contexts}
                    if not verdict.evidence or any(
                        ref not in valid_ids for ref in verdict.evidence
                    ):
                        raise ValueError("judge evidence is absent or not among supplied contexts")
                revisions.add(verdict.revision)
                tokens += verdict.tokens
                values.append(verdict.score)
                points.append(
                    Point(
                        id=case.id,
                        metric=name,
                        value=verdict.score,
                        reason=verdict.reason,
                        evidence=verdict.evidence,
                    )
                )
            except (ValueError, TypeError, KeyError, RuntimeError, OSError, httpx.HTTPError) as exc:
                failures += 1
                points.append(
                    Point(id=case.id, metric=name, status="judge_error", reason=type(exc).__name__)
                )
        n_total = len(batch.cases)
        n_missing = sum(p.status == "missing_input" for p in points)
        count = len(values)
        status = (
            "complete" if count == n_total and n_total else "partial" if count else "judge_error"
        )
        return Score(
            name=name,
            value=sum(values) / count if count else None,
            status=status,
            n_total=n_total,
            n_eligible=n_total - n_missing - failures,
            n_scored=count,
            n_failed=failures,
            n_missing=n_missing,
            source="judge",
            version=rubric.version,
            unit="rubric",
            points=points,
            details={
                "rubric_id": rubric.id,
                "rubric_hash": digest(rubric.model_dump(mode="json")),
                "model": judge.model,
                "provider": judge.provider,
                "method": "LLM-as-judge; estimated, not ground truth",
                "budget_calls": budget.calls,
                "reserved_tokens": budget.tokens,
                "reported_tokens": tokens,
                "judge_revisions": sorted(revisions),
            },
        )
