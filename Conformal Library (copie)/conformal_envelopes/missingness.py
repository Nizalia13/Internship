# HOW DO WE TRANSFORM SCORES AND HANDLE THE MISSING VALUES

from __future__ import annotations

import numpy as np


def missingness_mask(values) -> np.ndarray:
    """
       Return 1 where a score is observed and 0 where it is NaN.
    """
    arr = np.asarray(values, dtype=float)
    return (~np.isnan(arr)).astype(np.uint8)

##############################################################################################

def transform_scores(values, score_direction="higher_is_better") -> np.ndarray:
    """
        Convert raw predictor scores to nonconformity scores while preserving NaNs.

    Parameters
    ----------
    score_direction:
        "higher_is_better"
            Uses v = 1 - score, assumes scores are on [0, 1].

        "lower_is_better"
            Uses v = score. Use this when lower raw values already mean greater conformity.

        callable
            a custom NumPy-compatible transformation. The callable must preserve shape. 
            NaNs from the input are restored afterward.
    """


    arr = np.asarray(values, dtype=float)
    if np.isinf(arr).any():
        raise ValueError("Scores cannot contain +inf or -inf; use NaN for missing values.")

    # handling custom functions - ensures that the function returns the exact same shape dimension as input
    if callable(score_direction):
        out = np.asarray(score_direction(arr.copy()), dtype=float)
        if out.shape != arr.shape:
            raise ValueError("A custom score transformation must preserve shape.")
        # ensures that any position that was NaN remains NaN
        out[np.isnan(arr)] = np.nan
        return out

    # Case1: when raw score is a probability for we do 1-score
    if score_direction == "higher_is_better":
        finite = arr[np.isfinite(arr)]
        if len(finite) and ((finite < 0).any() or (finite > 1).any()):
            raise ValueError(
                "'higher_is_better' uses 1-score and therefore requires finite "
                "scores in [0, 1]. Use 'lower_is_better' or a custom transform "
                "for another score scale."
            )
        return np.where(np.isnan(arr), np.nan, 1.0 - arr)

    # when raw score is a error metric, we just keep it as is
    if score_direction == "lower_is_better":
        return arr.copy()

    raise ValueError("score_direction must be 'higher_is_better'," "'lower_is_better', or a callable.")


######################################################################################################

def nanmean_rows(values) -> np.ndarray:
    """
       Compute the mean of each sample over its observed score dimensions.

       NaNs are ignored. Fully missing rows remain NaN.
       This helper is mainly used by the collapsed envelope method.
    """
    arr = np.asarray(values, dtype=float)

    if arr.ndim != 2:
        raise ValueError("Expected a 2D array: rows are samples, columns are scores.")

    # counting the observed (non NaN) enteries per row
    observed = ~np.isnan(arr)
    counts = observed.sum(axis=1)
    sums = np.nansum(arr, axis=1)

    
    out = np.full(arr.shape[0], np.nan)
    np.divide(sums, counts, out=out, where=counts > 0)
    return out
