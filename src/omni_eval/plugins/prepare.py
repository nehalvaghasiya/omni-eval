"""No silent installation of unverified optional frameworks."""

from omni_eval.plugins.catalog import Catalog


class Preparer:
    def check(self, name: str) -> str:
        info = Catalog().get(name)
        return str(info["status"])

    def prepare(self, name: str, *, offline: bool = False) -> str:
        self.check(name)
        if name == "fixture":
            return "ready: dependency-free synthetic worker"
        raise RuntimeError(
            f"{name} integration has no qualified version/lock; do not install an unpinned framework"
        )
