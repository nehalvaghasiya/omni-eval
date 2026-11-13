from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TimingRecord(BaseModel):
    """Never confuse externally reported timing with OmniEval measurements."""

    model_config = ConfigDict(extra="forbid")
    latency_ms: float = Field(ge=0)
    provenance: Literal["measured_external", "measured_by_omnieval"] = (
        "measured_external"
    )
    ttft_ms: float | None = Field(default=None, ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
