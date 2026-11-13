from omni_eval.exceptions.omni_eval_error import OmniEvalError


class DependencyUnavailable(OmniEvalError):
    """A selected integration has not been installed."""
