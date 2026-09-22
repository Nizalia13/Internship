"""
Conformal Envelopes.

ConformalSetModel
    Fit one NaN-aware envelope per class and return conformal prediction sets.

evaluate_prediction_sets
    Compute basic set-valued classification metrics.
"""

from .model import ConformalSetModel
from .metrics import evaluate_prediction_sets


ConformalEnvelopeClassifier = ConformalSetModel

__all__ = ["ConformalSetModel", "ConformalEnvelopeClassifier", "evaluate_prediction_sets",]

__version__ = "0.2.0"
