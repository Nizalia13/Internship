from __future__ import annotations

import numpy as np


def conformal_quantile(values, alpha: float) -> float:
    """Finite-sample split-conformal quantile.

    For n calibration values, this returns the
    ceil((n+1)(1-alpha))-th order statistic, clipped to the available range.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        raise ValueError("Cannot calibrate from an empty set of finite values.")

    values = np.sort(values)
    n = len(values)
    idx = int(np.ceil((n + 1) * (1.0 - alpha))) - 1
    return float(values[np.clip(idx, 0, n - 1)])


def quantile_for_class(tau_scores, labels, cls, alpha: float) -> float:
    """Notebook-compatible class-conditional quantile."""
    tau_scores = np.asarray(tau_scores, dtype=float)
    labels = np.asarray(labels)
    vals = np.sort(tau_scores[labels == cls])

    if len(vals) == 0:
        return 0.0

    n = len(vals)
    idx = int(np.ceil((n + 1) * (1.0 - alpha))) - 1
    return float(vals[np.clip(idx, 0, n - 1)])


def compute_t_hat_multiclass(tau_scores, labels, classes, alpha: float) -> float:
    """Maximum of the class-conditional conformal quantiles.

    This mirrors the helper used in the supplied cleaned notebook.
    """
    return max(
        quantile_for_class(tau_scores, labels, cls, alpha)
        for cls in classes
    )
