import json
from pathlib import Path

from pydantic import ValidationError

from omni_eval.exceptions import InputError
from omni_eval.runners.runner_manifest import RunnerManifest


class RunnerResultImporter:
    """Import recorded versioned runner evidence without runner installation."""

    def import_file(self, path: str | Path) -> RunnerManifest:
        try:
            result = RunnerManifest.model_validate(
                json.loads(Path(path).read_text(encoding="utf-8"))
            )
        except (OSError, ValueError, ValidationError) as exc:
            raise InputError(f"invalid runner result: {exc}") from exc

        if result.schema_version != "1.0":
            raise InputError("unsupported external runner result schema version")

        return result
