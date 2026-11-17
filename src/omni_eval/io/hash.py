"""Stable hashes for source and normalized configuration."""

import hashlib
import json
from pathlib import Path
from typing import Any


def digest(data: Any) -> str:
    blob = json.dumps(
        data,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        default=str,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(blob).hexdigest()


def hash_file(path: str | Path) -> str:
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(262144), b""):
            sha.update(chunk)
    return sha.hexdigest()
