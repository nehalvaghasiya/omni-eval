"""Asset reference, no automatic network fetching."""

from pydantic import BaseModel, ConfigDict, Field


class Asset(BaseModel):
    model_config = ConfigDict(extra="forbid")
    uri: str = Field(min_length=1)
    mime: str
    sha256: str | None = None
    size: int | None = Field(default=None, ge=0)
