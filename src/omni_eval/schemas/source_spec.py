from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from omni_eval.schemas.target_spec import TargetSpec


class SourceSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["offline", "endpoint", "hybrid", "replay"] = "offline"
    predictions: str | None = None
    target: TargetSpec | None = None
    replay_path: str | None = None

    @model_validator(mode="after")
    def check_mode(self) -> "SourceSpec":
        if self.mode == "offline" and not self.predictions:
            raise ValueError("offline source requires predictions path")

        if self.mode in ("endpoint", "hybrid") and self.target is None:
            raise ValueError("endpoint/hybrid source requires a target")

        if self.mode == "hybrid" and not self.predictions:
            raise ValueError("hybrid source requires saved baseline predictions")

        if self.mode == "replay" and not self.replay_path:
            raise ValueError("replay source requires replay_path")

        return self
