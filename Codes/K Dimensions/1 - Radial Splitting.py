import numpy as np

def simulate_scores(n_phages, K, mean = None, correlation = 0.0, variance = 0.02):
    """
    Simulates K-dimensional conformal scores
    mean = array of length K (specifies the mean for all directions)
    """

    if mean is None:
        mean = np.full(K, 0.3)   # default mean for all dimensions
    cov = np.full((K, K), correlation * variance)
    np.fill_diagonal(cov, variance)
    scores = np.abs(np.random.multivariate_normal(mean, cov, n_phages))

    return np.clip(scores, 0 , 1)

def split_data(scores, split_ratio=0.5):
    """
    Randomly splits scores into S1 (shape discovery) and S2 (size scaling).
    """

    n = len(scores)
    idx = np.random.permutation(n)   
    cut = int(n * split_ratio)   # index to split the data
    return scores[idx[:cut]], scores[idx[cut:]]    

def sample_positive_sphere(M, K):
    """
    Samples M directions uniformly on the positive orthant of S^(K-1).
    Method: draw from N(0,I), take absolute values, normalise.
    """

    V = np.abs(np.random.randn(M, K))           # (M, K) — raw positive vectors
    norms = np.linalg.norm(V, axis=1, keepdims=True)    # norms of each vector
    return V / (norms + 1e-12)

def shape_discovery(S1, alpha, M, delta_deg=10):
    """
    For each of M directions u_m on S^(K-1)_+ it will:
      1. Find calibration points whose direction is within delta_deg of u_m
      2. Compute the (1-alpha) quantile of their magnitudes (gives a local radius threshold q_tilde[m])
    """

    K = S1.shape[1]     
    U = sample_positive_sphere(M, K)             # (M, K)

    # Normalizing S1 rows to get their directions
    mags = np.linalg.norm(S1, axis=1)            # magnitudes of S1 points
    dirs = S1 / (mags[:, None] + 1e-12)          # directions of the S1 points

    cos_thresh = np.cos(np.radians(delta_deg))    # threshold for being close to u_m

    q_tilde = np.zeros(M)
    for m in range(M):
        # Cosine similarity between each S1 point's direction and u_m
        sims = dirs @ U[m]                       # cosine similarities of all points with u_m
        mask = sims >= cos_thresh                # points that are withing the delta_deg of u_m

        if mask.sum() < 5:
            # If there are very few points near this direction - fall back to global quantile
            q_tilde[m] = np.quantile(mags, 1 - alpha)
        else:
            q_tilde[m] = np.quantile(mags[mask], 1 - alpha)

    return U, q_tilde      


def size_scaling(S2, U, q_tilde, alpha):
    """
    Finds t_hat: the (1-alpha) quantile of t*(s) over S2, where
        t*(s) = min_m { max_k (s_k / boundary_k_m) }
    that is the smallest scale factor t such that s lies inside
    """

    # Boundary point for each direction: b_m = u_m * q_tilde[m]
    boundary = U * q_tilde[:, None]              # calculating the boundary points for each region

    # For each S2 point and each sector m, we need to find out how much would we need to scale the boundary to just reach s?
    # ratio[i, m, k] = S2[i,k] / boundary[m,k]
    # t for sector m on point i = max over k 
    # t*(s_i) = min over m (only need one sector to cover it)

    ratios = S2[:, None, :] / (boundary[None, :, :] + 1e-12)    # how much to scale each boundary point to reach S2[i]
    t_per_sector = ratios.max(axis=2)            # max over k to get t for each sector m on point i
    tau_scores = t_per_sector.min(axis=1)        # min over m to get t*(si)

    n2 = len(tau_scores)
    idx = int(np.ceil((n2 + 1) * (1 - alpha))) - 1    # index ffor 1-alpha quantile
    t_hat = np.sort(tau_scores)[np.clip(idx, 0, n2 - 1)]   
    return t_hat, tau_scores

def is_in_region(scores, U, q_tilde, t_hat):
    """
    Returns boolean array: True if the score vector lies inside the final scaled envelope
    """

    q_final = q_tilde * t_hat     # scaling the quantiles to get the final boundary points
    boundary = U * q_final[:, None]            
    # A point is inside if any sector m covers it (all K dims ≤ boundary)
    inside = np.any(np.all(scores[:, None, :] <= boundary[None, :, :], axis=2), axis=1)

    return inside

def evaluate_coverage(S2, U, q_tilde, t_hat):
    """Empirical coverage on the size-scaling set."""
    mask = is_in_region(S2, U, q_tilde, t_hat)
    return mask.mean()

np.random.seed(42)
alpha = 0.1
M = 200

for K in [2, 5, 10, 20]:
    scores = simulate_scores(n_phages=1000, K=K, correlation=0.3)
    S1, S2 = split_data(scores, split_ratio=0.5)

    U, q_tilde = shape_discovery(S1, alpha, M, delta_deg=15)
    t_hat, _ = size_scaling(S2, U, q_tilde, alpha)
    coverage = evaluate_coverage(S2, U, q_tilde, t_hat)

    print(f"K={K:2d} | t_hat={t_hat:.3f} | empirical coverage={coverage:.3f} " f"(target={1-alpha:.2f})")