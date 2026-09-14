### DURING TESTING
### new score vector --> mean over observed dimensions --> one collapsed value
###        --> compare with q_tilde * t_hat --> inside/outside this label's envelope


### DURING TESTING
### new score vector --> mean observed dims --> one collapsed value --> comapre with q tilde x t hat --> predict

from __future__ import annotations

import numpy as np

from ..calibration import conformal_quantile
from ..missingness import nanmean_rows

EPS = 1e-12

# Takes the matrix of scores --> averages each row over observed dimensions 
#       --> check if every row is usable  --> return 1D values 
def _checked_row_means(values, stage: str) -> np.ndarray:
    """
       Average observed dimensions; flag rows with no usable mean.
    """
    # taking the mean using the fucntion from missingness.py (ignoring NaNs)
    means = nanmean_rows(values)

    # for empty datasets
    if means.size == 0:
        raise ValueError(f"{stage}: no samples were supplied.")

    # If there is an invalid row (no score for a the given label)
    invalid = np.flatnonzero(~np.isfinite(means))
    if invalid.size:
        raise ValueError(
            f"{stage}: rows {invalid.tolist()} have an undefined or non-finite mean. Each sample needs at least one observed "
            "finite score for this label."
        )
    # returns the collapsed 1D scores
    return means


#################################################################################################################################3

def build_collapsed(S1, S2, alpha: float) -> dict:
    """ 
       Learn the collapsed shape on S1 and calibrate its scale on S2.
    """

    # takes all the multidimensional score vectors and collapses them to one number
    S1_1d = _checked_row_means(S1, "Shape discovery (S1)")
    S2_1d = _checked_row_means(S2, "Calibration (S2)")

    # calculates the shape boundary 
    q_tilde = float(np.nanquantile(S1_1d, 1.0 - alpha))

    # how much would we have to scale the boundary 
    tau_scores = S2_1d / (q_tilde + EPS)

    # takes all the scaling factors and finds the conformal threshold
    t_hat = conformal_quantile(tau_scores, alpha)

    return {"method": "collapsed", "q_tilde": q_tilde, "t_hat": t_hat,}


#################################################################################################################################3

# used to evaluate new observations relative to the fitted envelope
def collapsed_tau(scores, envelope: dict) -> np.ndarray:
    """
       Return scores normalized by the calibrated boundary.
    """
    scores_1d = _checked_row_means(scores, "Prediction")
    q_tilde = float(envelope["q_tilde"])
    t_hat = float(envelope["t_hat"])

    # Avoid 0 * infinity when calibration requires an infinite threshold.
    if np.isposinf(t_hat):
        return np.zeros_like(scores_1d)

    # gives the position relative to the boundary
    return scores_1d / (q_tilde * t_hat + EPS)

#################################################################################################################################3

# used to check if the new point is inside the envelope or not
def collapsed_is_in_region(scores, envelope: dict) -> np.ndarray:
    """Use the host notebook's direct boundary comparison."""
    scores_1d = _checked_row_means(scores, "Prediction")
    q_tilde = float(envelope["q_tilde"])
    t_hat = float(envelope["t_hat"])

    if np.isposinf(t_hat):
        return np.ones(scores_1d.shape, dtype=bool)

    return scores_1d <= q_tilde * t_hat
