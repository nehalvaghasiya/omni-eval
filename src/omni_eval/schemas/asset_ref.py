from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, field_validator


class AssetRef(BaseModel):
    """An asset reference; loading or downloading is never implicit."""

    model_config = ConfigDict(extra="forbid")
    uri: str
    media_type: str
    sha256: str | None = None

    @field_validator("uri")
    @classmethod
    def safe_uri(cls, value: str) -> str:
        parsed = urlparse(value)

        if (
            not value
            or parsed.scheme not in ("", "https", "s3")
            or (parsed.scheme and not parsed.netloc)
        ):
            raise ValueError("asset URI must be a relative path, HTTPS or S3 reference")

        if ".." in parsed.path.split("/"):
            raise ValueError("asset URI cannot contain parent traversal")

        return value
