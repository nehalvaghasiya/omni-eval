"""Resolve presets and prerequisites before model/network acquisition."""

from omni_eval.core.registry import Registry
from omni_eval.types.case import Case
from omni_eval.types.config import Config

_PRESETS = {
    "classification": ["accuracy", "macro_f1"],
    "vision": ["accuracy"],
    "text": ["exact", "token_f1"],
    "retrieval": ["hit@5", "recall@5", "mrr"],
    "rag": ["answer_match", "citation_valid"],
    "agent": ["state_success", "tool_allowed"],
    "mcp": ["state_success", "tool_allowed"],
    "audio": ["wer", "cer"],
}


class Planner:
    def build(self, config: Config, cases: list[Case]) -> list[str]:
        names = config.metrics or _PRESETS.get(config.task)
        if not names:
            raise ValueError(
                f"task {config.task!r} has no safe default metrics; set metrics explicitly"
            )
        names = list(names)
        if len(names) != len(set(names)):
            raise ValueError("duplicate metric names in configuration")
        for name in names:
            if ":" in name:
                from omni_eval.plugins.catalog import Catalog

                provider = name.split(":", 1)[0]
                Catalog().require(provider, name)
                if provider == "judge":
                    from omni_eval.plugins.judge import JudgePlugin

                    JudgePlugin().validate(name, config)
                continue
            metric = Registry().get(name)
            if config.task not in metric.tasks:
                raise ValueError(f"metric {name!r} is not valid for task {config.task!r}")
            reference_needs = [
                x
                for x in metric.needs
                if x == "expected" or x.startswith("expected.") or x == "label_truth"
            ]
            if (
                reference_needs
                and cases
                and all(not all(metric._has(c, n) for n in reference_needs) for c in cases)
            ):
                raise ValueError(
                    f"Cannot compute {name}: no cases have required references {reference_needs}"
                )
        return names
