from __future__ import annotations

import numpy as np


def missingness_mask(values) -> np.ndarray:
    """Return 1 where a score is observed and 0 where it is NaN."""
    arr = np.asarray(values, dtype=float)
    return (~np.isnan(arr)).astype(np.uint8)


def transform_scores(values, score_direction="higher_is_better") -> np.ndarray:
    """Convert raw predictor scores to nonconformity scores while preserving NaNs.

    Parameters
    ----------
    score_direction:
        "higher_is_better"
            Uses v = 1 - score. This is the transformation used in the supplied
            host-classification notebook and assumes scores are on [0, 1].

        "lower_is_better"
            Uses v = score. Use this when lower raw values already mean greater
            conformity.

        callable
            Advanced use: a custom NumPy-compatible transformation. The callable
            must preserve shape. NaNs from the input are restored afterward.
    """
    arr = np.asarray(values, dtype=float)

    if callable(score_direction):
        out = np.asarray(score_direction(arr.copy()), dtype=float)
        if out.shape != arr.shape:
            raise ValueError("A custom score transformation must preserve shape.")
        out[np.isnan(arr)] = np.nan
        return out

    if score_direction == "higher_is_better":
        finite = arr[np.isfinite(arr)]
        if len(finite) and ((finite < 0).any() or (finite > 1).any()):
            raise ValueError(
                "'higher_is_better' uses 1-score and therefore requires finite "
                "scores in [0, 1]. Use 'lower_is_better' or a custom transform "
                "for another score scale."
            )
        return np.where(np.isnan(arr), np.nan, 1.0 - arr)

    if score_direction == "lower_is_better":
        return arr.copy()

    raise ValueError(
        "score_direction must be 'higher_is_better', "
        "'lower_is_better', or a callable."
    )


def nanmean_rows(values, all_missing_value: float = 0.0) -> np.ndarray:
    """NaN-aware row mean without warnings on fully missing rows."""
    arr = np.asarray(values, dtype=float)
    observed = ~np.isnan(arr)
    counts = observed.sum(axis=1)
    sums = np.nansum(arr, axis=1)

    out = np.full(arr.shape[0], float(all_missing_value))
    np.divide(sums, counts, out=out, where=counts > 0)
    return out
