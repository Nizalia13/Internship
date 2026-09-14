### S1 --> learn the shape of the radial boundary
### S2 --> learn how much that shape must be scaled conformally
### test point --> calculate its magnitude and direction
#              --> find the learned boundary radius in that direction
#              --> compare magnitude with radius × t_hat

from __future__ import annotations

import numpy as np

from ..calibration import conformal_quantile

EPS = 1e-12

def _checked_scores(values, stage: str) -> np.ndarray:
    """
       Validate nonconformity scores while allowing partial NaNs.
    """

    # converts whatever the user supplied into a NumPy floating-point array.
    arr = np.asarray(values, dtype=float)

    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{stage}: expected a nonempty 2D score array.")

    observed = arr[~np.isnan(arr)]
    if np.isinf(observed).any() or (observed < 0).any():
        raise ValueError(f"{stage}: observed nonconformity scores must be finite and nonnegative.")

    # if there is any row wit only NaNs
    missing_rows = np.flatnonzero(np.isnan(arr).all(axis=1))
    if missing_rows.size:
        raise ValueError(f"{stage}: rows {missing_rows.tolist()} have no observed scores for this label.")

    # if everything is valid we get an array
    return arr

#################################################################################################################################3

# generates the reference directions used to construct the radial envelope.
def sample_positive_sphere(rng, M: int, K: int) -> np.ndarray:
    """
       Sample M unit directions in the postive orthant of a K dimesnsional space.
    """
    V = np.abs(rng.standard_normal((M, K)))
    return V / (np.linalg.norm(V, axis=1, keepdims=True) + EPS)


#################################################################################################################################3

# calculates the radial boundary for one or more directions
def radial_boundary_radius(dirs_query, U, q_tilde, smoothing: float = 8.0):
    """
       Smoothly blend sampled radii using directional cosine similarity.
       - dirs_query contains the direction of the point to be evaluated.
       - U contains the reference directions.
       - q_tilde contains the learned radius associated with every reference direction.
    """
    # similarity between new and refernece directions
    cos_sim = dirs_query @ U.T

    # converts similarities into weights, directions closer to the new one get higher weight
    weights = np.exp(smoothing * (cos_sim - cos_sim.max(axis=1, keepdims=True)))
    weights /= weights.sum(axis=1, keepdims=True)
    # gives us a smooth radial boundary instead of random jumps
    return weights @ q_tilde


#################################################################################################################################3

# takes the multidimensional score vector and seperates it into magnitude and direction
def _magnitudes_and_directions(S):
    S = np.asarray(S, dtype=float)
    N, K = S.shape
    mags = np.zeros(N)
    dirs = np.zeros((N, K))

    for n in range(N):
        # which dimensions are observed (skip row if 0 observed)
        valid = ~np.isnan(S[n])
        if not valid.any():
            continue
        # calculates the euclidean magnitude using the observed dimensions
        mags[n] = np.linalg.norm(S[n, valid])
        if mags[n] > EPS:
            dirs[n, valid] = S[n, valid] / mags[n]

    return mags, dirs


#################################################################################################################################3

# constructs one radial envelope
def build_radial(S1, S2, alpha: float, *, n_directions: int = 250, neighbor_fraction: float = 0.2,
                 rng, smoothing: float = 8.0, angle_deg: float = 30.0,) -> dict:
    """
       Build the NaN-aware radial envelope from the cleaned notebook.
    """

    S1 = _checked_scores(S1, "Shape discovery (S1)")
    S2 = _checked_scores(S2, "Calibration (S2)")
    N1, K = S1.shape

    if S2.shape[1] != K:
        raise ValueError("S1 and S2 must have the same number of score columns.")

    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between 0 and 1.")

    if (isinstance(n_directions, (bool, np.bool_)) or not isinstance(n_directions, (int, np.integer)) or n_directions < 1):
        raise ValueError("n_directions must be a positive integer.")

    if not np.isfinite(neighbor_fraction) or not 0 < neighbor_fraction <= 1:
        raise ValueError("neighbor_fraction must be in (0, 1].")

    if not np.isfinite(smoothing) or smoothing < 0:
        raise ValueError("smoothing must be finite and nonnegative.")

    if not np.isfinite(angle_deg) or not 0 < angle_deg <= 90:
        raise ValueError("angle_deg must be in (0, 90].")

    if rng is None:
        raise ValueError("Provide a random generator for reproducibility.")

    # generates the refernece directions
    U = sample_positive_sphere(rng, int(n_directions), K)
    mags, dirs = _magnitudes_and_directions(S1)
    # creates the missingness mask
    observed = ~np.isnan(S1)

    q_tilde = np.zeros(int(n_directions))
    marginal_mag_q = float(np.quantile(mags, 1.0 - alpha))
    min_neighbors = max(10, int(float(neighbor_fraction) * N1))
    cos_band = np.cos(np.deg2rad(float(angle_deg)))

    for m in range(int(n_directions)):
        # Norm of this reference direction over each sample's observed coordinates.
        reference_norms = np.sqrt(np.sum(observed * U[m] ** 2, axis=1))

        # Compare directions in the same observed-coordinate subspace. 
        cos_sims = (dirs @ U[m]) / (reference_norms + EPS)
        in_band = cos_sims >= cos_band
        n_in_band = int(in_band.sum())

        if n_in_band >= min_neighbors:
            q_tilde[m] = np.quantile(mags[in_band], 1.0 - alpha)
        elif n_in_band > 0:
            w = n_in_band / min_neighbors
            local_q = np.quantile(mags[in_band], 1.0 - alpha)
            q_tilde[m] = (w * local_q + (1.0 - w) * marginal_mag_q)
        else:
            q_tilde[m] = marginal_mag_q

    unit_envelope = {"method": "radial", "U": U, "q_tilde": q_tilde, "t_hat": 1.0, "smoothing": float(smoothing),
                     "angle_deg": float(angle_deg), "n_directions": int(n_directions),
                     "neighbor_fraction": float(neighbor_fraction),}
    
    raw_tau = radial_tau(S2, unit_envelope)
    unit_envelope["t_hat"] = conformal_quantile(raw_tau, alpha)
    return unit_envelope


#################################################################################################################################3

# evaluates the new points against already trained radial envelope
def radial_tau(scores, envelope: dict) -> np.ndarray:
    U = envelope["U"]
    q_tilde = envelope["q_tilde"]
    t_hat = float(envelope["t_hat"])
    smoothing = float(envelope.get("smoothing", 8.0))

    scores = _checked_scores(scores, "Radial score evaluation")

    if scores.shape[1] != U.shape[1]:
        raise ValueError("Scores must match the envelope's number of columns.")

    N = scores.shape[0]
    tau = np.zeros(N)

    # Avoid radius * infinity, particularly when the radius is zero.
    if np.isposinf(t_hat):
        return tau

    for n in range(N):
        valid = ~np.isnan(scores[n])

        mag = np.linalg.norm(scores[n, valid])
        if mag < EPS:
            continue

        direction = scores[n, valid] / mag

        U_valid = U[:, valid]
        U_valid = U_valid / (np.linalg.norm(U_valid, axis=1, keepdims=True) + EPS)
            

        radius = radial_boundary_radius(direction[None, :], U_valid, q_tilde, smoothing,)[0]
        tau[n] = mag / (radius * t_hat + EPS)

    return tau
#################################################################################################################################3

def radial_is_in_region(scores, envelope: dict) -> np.ndarray:
    return radial_tau(scores, envelope) <= 1.0
