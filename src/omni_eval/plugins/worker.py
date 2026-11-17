"""Dependency-free synthetic worker for testing only."""

import sys

from omni_eval.plugins.reply import WorkerReply
from omni_eval.plugins.request import WorkerRequest


def main() -> None:
    request = WorkerRequest.model_validate_json(sys.stdin.read())
    if request.provider != "fixture" or request.metrics != ["fixture:echo"]:
        raise ValueError("fixture worker only handles fixture:echo")
    size = len(request.cases)
    reply = WorkerReply(
        job=request.job,
        provider="fixture",
        status="complete",
        metrics=[
            {
                "name": "fixture:echo",
                "value": 1.0 if size else None,
                "n_total": size,
                "n_eligible": size,
                "n_scored": size,
                "unit": "fraction",
                "source": "fixture",
                "details": {"synthetic": True},
            }
        ],
        provenance={"synthetic": True, "protocol": 1},
    )
    sys.stdout.write(reply.model_dump_json())


if __name__ == "__main__":
    main()
