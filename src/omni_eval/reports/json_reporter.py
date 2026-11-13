from omni_eval.schemas.run_result import RunResult


class JsonReporter:
    def render(self, run: RunResult) -> str:
        return run.model_dump_json(indent=2)
