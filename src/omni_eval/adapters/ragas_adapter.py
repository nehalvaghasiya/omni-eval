from omni_eval.exceptions import DependencyUnavailable


class RagasAdapter:
    """Optional dependency boundary; upstream metric definitions stay distinct."""

    def verify_available(self) -> str:
        try:
            import importlib.metadata

            return importlib.metadata.version("ragas")
        except importlib.metadata.PackageNotFoundError as exc:
            raise DependencyUnavailable(
                "install omni-eval[ragas] in an isolated environment"
            ) from exc
