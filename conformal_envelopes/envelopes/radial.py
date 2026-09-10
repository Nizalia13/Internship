from __future__ import annotations

import numpy as np

from ..calibration import conformal_quantile

EPS = 1e-12


def sample_positive_sphere(rng, M: int, K: int) -> np.ndarray:
    """Sample M unit directions in the positive orthant."""
    V = np.abs(rng.standard_normal((M, K)))
    return V / (np.linalg.norm(V, axis=1, keepdims=True) + EPS)


def radial_boundary_radius(dirs_query, U, q_tilde, smoothing: float = 8.0):
    """Smoothly blend sampled radii using directional cosine similarity.

    The notebook calls this parameter kappa. The public API calls it
    ``smoothing`` because that is easier for a library user to interpret.
    Higher values produce a sharper, more local boundary.
    """
    cos_sim = dirs_query @ U.T
    weights = np.exp(
        smoothing * (cos_sim - cos_sim.max(axis=1, keepdims=True))
    )
    weights /= weights.sum(axis=1, keepdims=True)
    return weights @ q_tilde


def _magnitudes_and_directions(S):
    S = np.asarray(S, dtype=float)
    N, K = S.shape
    mags = np.zeros(N)
    dirs = np.zeros((N, K))

    for n in range(N):
        valid = ~np.isnan(S[n])
        if not valid.any():
            continue

        mags[n] = np.linalg.norm(S[n, valid])
        if mags[n] > EPS:
            dirs[n, valid] = S[n, valid] / mags[n]

    return mags, dirs


def build_radial(
    S1,
    S2,
    alpha: float,
    *,
    n_directions: int = 250,
    neighbor_fraction: float = 0.2,
    rng,
    smoothing: float = 8.0,
    angle_deg: float = 30.0,
) -> dict:
    """Build the NaN-aware radial envelope from the cleaned notebook."""
    S1 = np.asarray(S1, dtype=float)
    N1, K = S1.shape

    if K == 0:
        raise ValueError("Radial envelope requires at least one score dimension.")

    U = sample_positive_sphere(rng, int(n_directions), K)
    mags, dirs = _magnitudes_and_directions(S1)

    q_tilde = np.zeros(int(n_directions))
    marginal_mag_q = float(np.quantile(mags, 1.0 - alpha))
    min_neighbors = max(10, int(float(neighbor_fraction) * N1))
    cos_band = np.cos(np.deg2rad(float(angle_deg)))

    for m in range(int(n_directions)):
        cos_sims = dirs @ U[m]
        in_band = cos_sims >= cos_band
        n_in_band = int(in_band.sum())

        if n_in_band >= min_neighbors:
            q_tilde[m] = np.quantile(mags[in_band], 1.0 - alpha)
        elif n_in_band > 0:
            w = n_in_band / min_neighbors
            local_q = np.quantile(mags[in_band], 1.0 - alpha)
            q_tilde[m] = (
                w * local_q + (1.0 - w) * marginal_mag_q
            )
        else:
            q_tilde[m] = marginal_mag_q

    unit_envelope = {
        "method": "radial",
        "U": U,
        "q_tilde": q_tilde,
        "t_hat": 1.0,
        "smoothing": float(smoothing),
        "angle_deg": float(angle_deg),
        "n_directions": int(n_directions),
        "neighbor_fraction": float(neighbor_fraction),
    }

    raw_tau = radial_tau(S2, unit_envelope)
    unit_envelope["t_hat"] = conformal_quantile(raw_tau, alpha)
    return unit_envelope


def radial_tau(scores, envelope: dict) -> np.ndarray:
    U = envelope["U"]
    q_tilde = envelope["q_tilde"]
    t_hat = float(envelope["t_hat"])
    smoothing = float(envelope.get("smoothing", 8.0))

    scores = np.asarray(scores, dtype=float)
    N = scores.shape[0]
    tau = np.zeros(N)

    for n in range(N):
        valid = ~np.isnan(scores[n])

        # Same rule as the cleaned notebook: a fully missing row contributes
        # no rejection signal.
        if not valid.any():
            tau[n] = 0.0
            continue

        mag = np.linalg.norm(scores[n, valid])
        if mag < EPS:
            tau[n] = 0.0
            continue

        direction = scores[n, valid] / mag

        U_valid = U[:, valid]
        U_valid /= (
            np.linalg.norm(U_valid, axis=1, keepdims=True) + EPS
        )

        radius = radial_boundary_radius(
            direction[None, :],
            U_valid,
            q_tilde,
            smoothing,
        )[0]

        tau[n] = mag / (radius * t_hat + EPS)

    return tau


def radial_is_in_region(scores, envelope: dict) -> np.ndarray:
    return radial_tau(scores, envelope) <= 1.0
