from __future__ import annotations

import numpy as np

from ..calibration import conformal_quantile

EPS = 1e-12


def strip_shape_discovery(
    S1,
    alpha: float,
    n_bins: int,
    *,
    smoothing_window: int = 2,
    min_samples: int = 3,
):
    """Discover NaN-aware conditional strip limits.

    This follows the supplied cleaned notebook. A small robustness guard is
    added for dimensions that are entirely NaN so they do not create invalid
    bin edges.
    """
    S1 = np.asarray(S1, dtype=float)
    _, K = S1.shape
    n_bins = int(n_bins)

    bin_edges = []
    marginal_quantiles = np.ones(K)

    for j in range(K):
        finite = S1[np.isfinite(S1[:, j]), j]

        if len(finite) == 0:
            bin_edges.append(np.linspace(0.0, 1.0, n_bins + 1))
            marginal_quantiles[j] = 1.0
            continue

        lo = float(np.min(finite))
        hi = float(np.max(finite))

        if np.isclose(lo, hi):
            pad = max(1e-6, abs(lo) * 1e-6)
            lo -= pad
            hi += pad

        bin_edges.append(np.linspace(lo, hi, n_bins + 1))
        marginal_quantiles[j] = np.quantile(
            finite, 1.0 - alpha
        )

    bin_edges = np.asarray(bin_edges)
    limits = np.zeros((K, K, n_bins))

    for j in range(K):
        for n in range(n_bins):
            lo, hi = bin_edges[j, n], bin_edges[j, n + 1]

            # The original notebook uses [lo, hi) for its bins.
            in_strip = (
                (~np.isnan(S1[:, j]))
                & (S1[:, j] >= lo)
                & (S1[:, j] < hi)
            )

            for i in range(K):
                if i == j:
                    continue

                src_mask = in_strip & (~np.isnan(S1[:, i]))
                n_src = int(src_mask.sum())

                non_nan_i = S1[~np.isnan(S1[:, i]), i]
                marginal_i = (
                    np.quantile(non_nan_i, 1.0 - alpha)
                    if len(non_nan_i) > 0
                    else marginal_quantiles[i]
                )

                if n_src >= min_samples:
                    limits[j, i, n] = np.quantile(
                        S1[src_mask, i], 1.0 - alpha
                    )
                elif n_src == 0:
                    limits[j, i, n] = marginal_i
                else:
                    w = n_src / min_samples
                    conditional_i = np.quantile(
                        S1[src_mask, i], 1.0 - alpha
                    )
                    limits[j, i, n] = (
                        w * conditional_i
                        + (1.0 - w) * marginal_i
                    )

    limits_raw = limits.copy()

    for j in range(K):
        for i in range(K):
            if i == j:
                continue
            for n in range(n_bins):
                hi_n = min(n + int(smoothing_window), n_bins - 1)
                limits[j, i, n] = np.max(
                    limits_raw[j, i, n:hi_n + 1]
                )

    return bin_edges, limits, marginal_quantiles


def get_bin_indices(scores, bin_edges):
    scores = np.asarray(scores, dtype=float)
    K = scores.shape[1]
    n_bins = bin_edges.shape[1] - 1

    return np.array([
        np.clip(
            np.where(
                np.isnan(scores[:, j]),
                0,
                np.digitize(scores[:, j], bin_edges[j]) - 1,
            ),
            0,
            n_bins - 1,
        )
        for j in range(K)
    ]).T


def strip_tau_scores(
    S,
    bin_edges,
    limits,
    marginal_quantiles,
):
    """Raw strip scaling score, preserving the notebook's NaN rules."""
    S = np.asarray(S, dtype=float)
    N, K = S.shape

    if K == 1:
        target_q = max(float(marginal_quantiles[0]), EPS)
        observed = np.where(np.isnan(S[:, 0]), 0.0, S[:, 0])
        return observed / target_q

    bin_idx = get_bin_indices(S, bin_edges)

    j_idx = np.arange(K)[None, :, None]
    i_idx = np.arange(K)[None, None, :]
    m_idx = bin_idx[:, :, None]

    lims = limits[j_idx, i_idx, m_idx]

    S_i = S[:, np.newaxis, :]
    S_j = S[:, :, np.newaxis]

    nan_i = np.isnan(S_i)
    nan_j = np.isnan(S_j)

    mq = marginal_quantiles[np.newaxis, np.newaxis, :]
    mq_i = np.broadcast_to(mq, (N, K, K))

    ratio = np.zeros((N, K, K))

    both_avail = (~nan_i) & (~nan_j)
    miss_i = nan_i & (~nan_j)
    miss_j = (~nan_i) & nan_j

    ratio = np.where(
        both_avail,
        np.broadcast_to(S_i, (N, K, K)) / (lims + EPS),
        ratio,
    )

    # Target i missing but conditioning j observed:
    # use the target's marginal quantile against the conditional limit.
    ratio = np.where(
        miss_i,
        mq_i / (lims + EPS),
        ratio,
    )

    # Conditioning j missing but target i observed:
    # fall back to the target's own marginal quantile.
    ratio = np.where(
        miss_j,
        np.broadcast_to(S_i, (N, K, K)) / (mq_i + EPS),
        ratio,
    )

    # Both missing stay at 0: no constraint is contributed.
    return np.nanmax(ratio, axis=(1, 2))


def build_strip(
    S1,
    S2,
    alpha: float,
    *,
    n_bins: int = 8,
    smoothing_window: int = 2,
    min_samples: int = 3,
) -> dict:
    bin_edges, limits, marginal_quantiles = strip_shape_discovery(
        S1,
        alpha,
        n_bins,
        smoothing_window=smoothing_window,
        min_samples=min_samples,
    )

    raw_tau = strip_tau_scores(
        S2,
        bin_edges,
        limits,
        marginal_quantiles,
    )

    t_hat = conformal_quantile(raw_tau, alpha)

    return {
        "method": "strip",
        "bin_edges": bin_edges,
        "limits": limits,
        "marginal_quantiles": marginal_quantiles,
        "t_hat": t_hat,
        "n_bins": int(n_bins),
        "smoothing_window": int(smoothing_window),
        "min_samples": int(min_samples),
    }


def strip_tau(scores, envelope: dict) -> np.ndarray:
    raw_tau = strip_tau_scores(
        scores,
        envelope["bin_edges"],
        envelope["limits"],
        envelope["marginal_quantiles"],
    )

    return raw_tau / (float(envelope["t_hat"]) + EPS)


def strip_is_in_region(scores, envelope: dict) -> np.ndarray:
    return strip_tau(scores, envelope) <= 1.0
