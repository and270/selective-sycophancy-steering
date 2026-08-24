"""Selective factual-sycophancy activation steering."""

from .api import SteeringTarget, resolve_steering_target, steer_model
from .hooks import SteeringAudit

__version__ = "1.0.0"

__all__ = [
    "SteeringAudit",
    "SteeringTarget",
    "__version__",
    "resolve_steering_target",
    "steer_model",
]
