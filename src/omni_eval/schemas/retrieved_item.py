from pydantic import BaseModel, ConfigDict, Field


class RetrievedItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    document_id: str
    chunk_id: str
    rank: int = Field(ge=1)
    score: float | None = None
    text: str | None = None
