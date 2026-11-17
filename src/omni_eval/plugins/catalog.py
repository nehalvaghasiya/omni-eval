"""Honest installed-wheel plugin support catalog."""

import json
from importlib.resources import files
from typing import Any, cast


class Catalog:
    def list(self) -> dict[str, dict[str, Any]]:
        return cast(
            dict[str, dict[str, Any]],
            json.loads(
                files("omni_eval.plugins").joinpath("catalog.json").read_text(encoding="utf-8")
            )["providers"],
        )

    def get(self, name: str) -> dict[str, Any]:
        providers = self.list()
        if name not in providers:
            raise ValueError(f"unknown plugin {name!r}")
        return providers[name]

    def require(self, name: str, metric: str | None = None) -> None:
        state = self.get(name)
        if state["status"] not in {"ready", "verified"}:
            raise RuntimeError(
                f"{name} is not prepared/qualified; real execution is unavailable; no native fallback"
            )
        if (
            name == "judge"
            and metric is not None
            and (not metric.startswith("judge:") or not metric.split(":", 1)[1])
        ):
            raise ValueError("judge metrics need an explicit rubric ID")
        if name == "fixture" and metric is not None and (metric != "fixture:echo"):
            raise ValueError("fixture only supports fixture:echo")
