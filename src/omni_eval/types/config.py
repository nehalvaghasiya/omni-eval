"""User-facing evaluation configuration."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from omni_eval.types.target import Target


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: int = 1
    name: str = "evaluation"
    task: str
    data: str | None = None
    predictions: str | None = None
    columns: dict[str, str] = Field(default_factory=dict)
    json_columns: list[str] = Field(default_factory=list)
    actual_source: str | None = None
    metrics: list[str] = Field(default_factory=list)
    source: str = "saved"
    target: Target | None = None
    judge: dict[str, Any] | None = None
    plugin: str | None = None
    suite: str | None = None
    output: dict[str, Any] = Field(default_factory=dict)
    gates: list[dict[str, Any]] = Field(default_factory=list)
    limits: dict[str, Any] = Field(default_factory=dict)
    slices: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_mode(self) -> "Config":
        if self.version != 1:
            raise ValueError("only config version 1 is supported")
        if self.source not in {"saved", "http", "openai", "replay"}:
            raise ValueError("source must be saved, http, openai or replay")
        if self.source in {"http", "openai"} and self.target is None:
            raise ValueError("live sources require an explicit target")
        if not self.data and self.source != "replay" and (not self.suite):
            raise ValueError("set data to a JSONL file (or configure suite)")
        if self.actual_source and self.actual_source != "predictions":
            raise ValueError("actual_source must be 'predictions'")
        return self

    @classmethod
    def from_file(cls, path: str | Path) -> "Config":
        file = Path(path)
        with file.open(encoding="utf-8") as stream:
            raw = yaml.safe_load(stream)
        if not isinstance(raw, dict):
            raise ValueError("config must contain a mapping")
        config = cls.model_validate(raw)
        return config.resolved(file.parent)

    def resolved(self, base: str | Path) -> "Config":
        data = self.model_dump()
        for key in ("data", "predictions", "suite"):
            if data.get(key) and (not Path(data[key]).is_absolute()):
                data[key] = str(Path(base) / data[key])
        judge = data.get("judge")
        if isinstance(judge, dict) and judge.get("rubric"):
            rubric = Path(judge["rubric"])
            if not rubric.is_absolute():
                judge["rubric"] = str(Path(base) / rubric)
        out = data.get("output", {})
        if isinstance(out, dict) and out.get("dir") and (not Path(out["dir"]).is_absolute()):
            out["dir"] = str(Path(base) / out["dir"])
        return Config.model_validate(data)
