from omni_eval.exceptions import DependencyUnavailable


class DeepEvalAdapter:
    """Optional dependency boundary; no silent alternative scoring semantics."""

    def verify_available(self) -> str:
        try:
            import importlib.metadata

            return importlib.metadata.version("deepeval")
        except importlib.metadata.PackageNotFoundError as exc:
            raise DependencyUnavailable(
                "install omni-eval[deepeval] in an isolated environment"
            ) from exc
