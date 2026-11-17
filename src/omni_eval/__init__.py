"""OmniEval: clear, reproducible AI evaluation from saved predictions."""

from importlib.metadata import PackageNotFoundError, version

from omni_eval.api import compare, evaluate, load, metrics
from omni_eval.types.case import Case
from omni_eval.types.run import Run

try:
    __version__ = version("omni-eval")
except PackageNotFoundError:
    __version__ = "0.2.0.dev0"
__all__ = ["evaluate", "load", "compare", "metrics", "Case", "Run", "__version__"]
