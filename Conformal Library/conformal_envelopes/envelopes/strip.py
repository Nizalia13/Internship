from __future__ import annotations
import numpy as np
from ..calibration import conformal_quantile

EPS = 1e-12


def strip_shape_discovery(S1, alpha: float, n_bins: int, *, smoothing_window: int = 2, min_samples: int = 3,):
    """
       Discover NaN-aware conditional strip limits.
    """
    S1 = np.asarray(S1, dtype=float)
    _, K = S1.shape
    n_bins = int(n_bins)

    bin_edges = []
    marginal_quantiles = np.ones(K)

    for j in range(K):
        finite = S1[np.isfinite(S1[:, j]), j]

        # if len(finite) == 0:
        #     bin_edges.append(np.linspace(0.0, 1.0, n_bins + 1))
        #     marginal_quantiles[j] = 1.0
        #     continue

        if len(finite) == 0:
            raise ValueError("All-NaN S1 columns must be removed before shape discovery.")

        lo = float(np.min(finite))
        hi = float(np.max(finite))

        if np.isclose(lo, hi):
            pad = max(1e-6, abs(lo) * 1e-6)
            lo -= pad
            hi += pad

        bin_edges.append(np.linspace(lo, hi, n_bins + 1))
        marginal_quantiles[j] = np.quantile(finite, 1.0 - alpha)

    bin_edges = np.asarray(bin_edges)
    limits = np.zeros((K, K, n_bins))

    for j in range(K):
        for n in range(n_bins):
            lo, hi = bin_edges[j, n], bin_edges[j, n + 1]

            upper_check = (S1[:, j] <= hi if n == n_bins - 1 else S1[:, j] < hi)
            in_strip = ((~np.isnan(S1[:, j])) & (S1[:, j] >= lo) & upper_check)

            for i in range(K):
                if i == j:
                    continue

                src_mask = in_strip & (~np.isnan(S1[:, i]))
                n_src = int(src_mask.sum())

                non_nan_i = S1[~np.isnan(S1[:, i]), i]
                marginal_i = (np.quantile(non_nan_i, 1.0 - alpha) if len(non_nan_i) > 0
                              else marginal_quantiles[i])

                # marginal_i = marginal_quantiles[i]

                if n_src >= min_samples:
                    limits[j, i, n] = np.quantile(S1[src_mask, i], 1.0 - alpha)

                elif n_src == 0:
                    limits[j, i, n] = marginal_i
                else:
                    w = n_src / min_samples
                    conditional_i = np.quantile(S1[src_mask, i], 1.0 - alpha)
                    limits[j, i, n] = (w * conditional_i + (1.0 - w) * marginal_i)

    limits_raw = limits.copy()

    for j in range(K):
        for i in range(K):
            if i == j:
                continue

            for n in range(n_bins):
                hi_n = min(n + int(smoothing_window), n_bins - 1)
                limits[j, i, n] = np.max(limits_raw[j, i, n:hi_n + 1])

    return bin_edges, limits, marginal_quantiles

#################################################################################################################################3


def get_bin_indices(scores, bin_edges):
    scores = np.asarray(scores, dtype=float)
    K = scores.shape[1]
    n_bins = bin_edges.shape[1] - 1

    return np.array([np.clip(np.where(np.isnan(scores[:, j]), 0, np.digitize(scores[:, j], bin_edges[j]) - 1,),
                             0, n_bins - 1,) for j in range(K)]).T

#################################################################################################################################3

def strip_tau_scores(S, bin_edges, limits, marginal_quantiles):
    """
       Evaluate constraints between distinct, observed coordinates.
    """
    S = np.asarray(S, dtype=float)

    if S.ndim != 2 or S.shape[0] == 0 or S.shape[1] == 0:
        raise ValueError("Expected a nonempty 2D score array.")

    N, K = S.shape
    if K != bin_edges.shape[0]:
        raise ValueError("Scores must match the envelope's number of columns.")

    observed = ~np.isnan(S)
    if np.isnan(S).all(axis=1).any():
        raise ValueError("A sample has no observed scores for this label.")

    if np.isinf(S[observed]).any() or (S[observed] < 0).any():
        raise ValueError("Observed scores must be finite and nonnegative.")

    bin_idx = get_bin_indices(S, bin_edges)
    tau = np.zeros(N)

    # With one observed coordinate, use its marginal limit.
    single_rows = np.flatnonzero(observed.sum(axis=1) == 1)

    if single_rows.size:
        single_cols = np.argmax(observed[single_rows], axis=1)
        tau[single_rows] = (S[single_rows, single_cols] / (marginal_quantiles[single_cols] + EPS))

    for j in range(K):
        for i in range(K):
            if i == j:
                continue

            valid = observed[:, j] & observed[:, i]
            local_limits = limits[j, i, bin_idx[valid, j]]
            ratios = S[valid, i] / (local_limits + EPS)
            tau[valid] = np.maximum(tau[valid], ratios)

    return tau

#################################################################################################################################3

def build_strip(S1, S2,  alpha: float, *, n_bins: int = 8, smoothing_window: int = 2, 
                min_samples: int = 3,) -> dict:
    """
         Learn the strip shape on S1 and calibrate its scale on S2.
    """
    S1 = np.asarray(S1, dtype=float)
    S2 = np.asarray(S2, dtype=float)

    if S1.ndim != 2 or S2.ndim != 2:
        raise ValueError("S1 and S2 must be 2D arrays.")

    if S1.shape[0] == 0 or S2.shape[0] == 0:
        raise ValueError("S1 and S2 must contain samples.")

    d = S1.shape[1]
    if d == 0 or S2.shape[1] != d:
        raise ValueError("S1 and S2 must have the same positive number of columns.")

    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between 0 and 1.")

    for stage, scores in [("S1", S1), ("S2", S2)]:
        observed = scores[~np.isnan(scores)]

        if np.isinf(observed).any() or (observed < 0).any():
            raise ValueError(f"{stage}: observed scores must be finite and nonnegative.")

        if np.isnan(scores).all(axis=1).any():
            raise ValueError(f"{stage}: a sample has no observed scores.")

    # Learn the retained columns from shape-discovery data only.
    keep_columns = np.flatnonzero(~np.isnan(S1).all(axis=0))
    n_input_features = S1.shape[1]

    if keep_columns.size == 0:
        raise ValueError("S1 has no observed score columns.")

    S1 = S1[:, keep_columns]
    S2 = S2[:, keep_columns]

    # Dropping columns can leave an individual sample entirely missing.
    for stage, scores in [("S1", S1), ("S2", S2)]:
        if np.isnan(scores).all(axis=1).any():
            raise ValueError(f"{stage}: a sample has no observed scores after column selection.")

    d = S1.shape[1]

    alpha_shape = alpha / d

    for name, value, minimum in [("n_bins", n_bins, 1), ("min_samples", min_samples, 1), 
                                 ("smoothing_window", smoothing_window, 0),]:
         if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) 
             or value < minimum):
             raise ValueError(f"{name} must be an integer >= {minimum}.")

    bin_edges, limits, marginal_quantiles = strip_shape_discovery(S1, alpha_shape, n_bins, 
                                                                  smoothing_window=smoothing_window,
                                                                  min_samples=min_samples,)

    raw_tau = strip_tau_scores(S2, bin_edges, limits, marginal_quantiles,)

    t_hat = conformal_quantile(raw_tau, alpha)

    return {"method": "strip", "bin_edges": bin_edges, "limits": limits, "marginal_quantiles": marginal_quantiles,
            "t_hat": t_hat, "n_bins": int(n_bins),"smoothing_window": int(smoothing_window),"min_samples": int(min_samples),
            "keep_columns": keep_columns, "n_input_features": n_input_features,}

#################################################################################################################################3

def _select_strip_columns(scores, envelope):
    scores = np.asarray(scores, dtype=float)

    if scores.ndim != 2:
        raise ValueError("Scores must be a 2D array.")

    if scores.shape[1] != envelope["n_input_features"]:
        raise ValueError("Prediction columns must match the original fit input.")

    return scores[:, envelope["keep_columns"]]

#################################################################################################################################3

def strip_tau(scores, envelope: dict) -> np.ndarray:
    scores = _select_strip_columns(scores, envelope)

    raw_tau = strip_tau_scores(scores, envelope["bin_edges"], envelope["limits"], 
                               envelope["marginal_quantiles"],)

    t_hat = float(envelope["t_hat"])

    if np.isposinf(t_hat):
        return np.zeros_like(raw_tau)

    if t_hat == 0.0:
        return np.where(raw_tau == 0.0, 0.0, np.inf)

    return raw_tau / t_hat

#################################################################################################################################3


def strip_is_in_region(scores, envelope: dict) -> np.ndarray:
    scores = _select_strip_columns(scores, envelope)
    raw_tau = strip_tau_scores(scores, envelope["bin_edges"], envelope["limits"],
                               envelope["marginal_quantiles"],)
    return raw_tau <= float(envelope["t_hat"])