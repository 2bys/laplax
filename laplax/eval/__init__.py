"""`eval` module contains pushforwards, calibration and metrics."""

from .calibrate import evaluate_for_given_prior_arguments, optimize_prior_prec
from .pushforward import set_lin_pushforward, set_nonlin_pushforward
from .utils import evaluate_metrics_on_dataset, evaluate_on_dataset

__all__ = [
    "evaluate_for_given_prior_arguments",
    "evaluate_metrics_on_dataset",
    "evaluate_on_dataset",
    "optimize_prior_prec",
    "set_lin_pushforward",
    "set_nonlin_pushforward",
]
