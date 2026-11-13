"""Content hashes never include file path metadata."""

import hashlib
import json
from pathlib import Path
from typing import Any


def hash_file(path: str | Path) -> str:
    digest = hashlib.sha256()

    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def hash_canonical(value: Any) -> str:
    data = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str, ensure_ascii=False
    )

    return hashlib.sha256(data.encode("utf-8")).hexdigest()
