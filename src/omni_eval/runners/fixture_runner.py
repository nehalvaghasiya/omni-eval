from omni_eval.runners.runner_manifest import RunnerManifest


class FixtureRunner:
    """Synthetic import-contract test; reports are not real benchmark scores."""

    def execute(self) -> RunnerManifest:
        return RunnerManifest(
            runner_id="fixture",
            runner_version="fixture-v1",
            suite_id="dry-run",
            status="completed",
            metrics={"synthetic_contract": 1.0},
            warnings=["DRY RUN / SYNTHETIC: not a benchmark"],
            is_fixture=True,
        )
