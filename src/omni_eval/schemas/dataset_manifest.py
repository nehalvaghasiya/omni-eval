from pydantic import BaseModel, ConfigDict, Field


class DatasetManifest(BaseModel):
    """Provenance of an immutable evaluation split."""

    model_config = ConfigDict(extra="forbid")
    schema_version: str = "1.0"
    dataset_id: str
    version: str
    task: str
    split: str
    sha256: str
    license: str | None = None
    extensions: dict[str, object] = Field(default_factory=dict)
