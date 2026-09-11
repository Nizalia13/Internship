# TURNS CALIBRATION NONCONFROMITY SCORES (TAU VALUES) INTO CONFORMAL THRESHOLDS (T_HAT)

from __future__ import annotations

import numpy as np

#  Getting the finite sample conformal threshold from calibration scores.
#  For n calibration values, this returns the ceil((n+1)(1-alpha))-th order statistic

def conformal_quantile(values, alpha: float) -> float:
    """
        Calibration scores --> sort them --> calculate conformal rank k --> take the k-th value --> conformal threshold t_hat = c

        - Input: 1D array of calibration scores and a significance level alpha in (0, 1).
        - It sorts the calibration scores and returns the finite-sample conformal quantile used to target 
          coverage of at least 1-alpha
        - Output: conformal threshold t_hat such that the conformal prediction set has coverage at least 1-alpha.
    """
    
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between 0 and 1.")

    values = np.asarray(values, dtype=float)

    if values.ndim != 1 or values.size == 0:
        raise ValueError("Calibration scores must be a nonempty 1D array.")

    # Preserve positive infinity; reject undefined or invalid scores.
    if np.isnan(values).any() or np.isneginf(values).any():
        raise ValueError("Calibration scores cannot contain NaN or -inf.")

    n = values.size
    k = int(np.ceil((n + 1) * (1 - alpha)))

    # Too few calibration observations for a finite threshold.
    if k > n:
        return float("inf")

    return float(np.sort(values)[k - 1])

##################################################################################

# Select one class's calibration scores and compute its threshold.

def quantile_for_class(tau_scores, labels, cls, alpha: float) -> float:
    """
        Compute the calibration threshold for one true class.
        Parameters:
           -  tau_scores: 1D array of calibration scores
           -  labels: 1D array of true class labels corresponding to tau_scores
           -  cls: the class for which to compute the threshold
           -  alpha: significance level for the conformal prediction set
        Returns:
           -  The conformal threshold for the specified class.
    """
    tau_scores = np.asarray(tau_scores, dtype=float)
    labels = np.asarray(labels)

    if tau_scores.ndim != 1 or labels.ndim != 1:
        raise ValueError("tau_scores and labels must be 1D arrays.")

    if tau_scores.size != labels.size:
        raise ValueError("Each calibration score must have one label.")

    # Check if the specified class is present in the labels
    class_scores = tau_scores[labels == cls]

    # prevents us from calibrating a class is there are no observations for it
    if class_scores.size == 0:
        raise ValueError(f"No calibration scores for class {cls!r}.")

    # Sends the class scores to the "conformal_quantile function" to get the threshold for the class
    return conformal_quantile(class_scores, alpha)


################################################################################


# def compute_t_hat_multiclass(tau_scores, labels, classes, alpha: float) -> float:
#     """
#        Maximum class specific threshold for a shared envelope.
#     """
#     classes = list(classes)

#     if not classes:
#         raise ValueError("At least one class is required.")

#     # If we use a shared envelope
#     return max(quantile_for_class(tau_scores, labels, cls, alpha) for cls in classes)
