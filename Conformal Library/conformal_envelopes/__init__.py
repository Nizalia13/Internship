# Makes the main classes and functions available through a import.
"""
Conformal Envelopes.

ConformalSetModel
    Fit one NaN-aware envelope per class and return conformal prediction sets.

evaluate_prediction_sets
    Compute basic set-valued classification metrics.
"""

from .model import ConformalSetModel
from .metrics import evaluate_prediction_sets
from .plotting import plot_inclusion_heatmap


ConformalEnvelopeClassifier = ConformalSetModel

__all__ = ["ConformalSetModel", "ConformalEnvelopeClassifier", 
           "evaluate_prediction_sets", "plot_inclusion_heatmap",]

__version__ = "0.2.0"
