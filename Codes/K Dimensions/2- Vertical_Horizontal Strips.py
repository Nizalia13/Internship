import numpy as np


# --- Data Simulation (K-dimensional) ---
def simulate_scores(n_phages, K, mean=None, correlation=0.0, variance=0.02):
    """
    Simulates K-dimensional conformal scores.
    mean = array of length K (specifies the mean for all directions)
    """
    if mean is None:
        mean = np.full(K, 0.3)   # default mean for all dimensions
    cov = np.full((K, K), correlation * variance)
    np.fill_diagonal(cov, variance)
    scores = np.abs(np.random.multivariate_normal(mean, cov, n_phages))

    return np.clip(scores, 0 , 1)


# --- Calibration Split ---
def split_data(scores, split_ratio=0.5):
    """
    Randomly splits scores into S1 (shape discovery) and S2 (size scaling).
    """

    n = len(scores)
    idx = np.random.permutation(n)   
    cut = int(n * split_ratio)   # index to split the data
    return scores[idx[:cut]], scores[idx[cut:]]   


# --- Shape Discovery (Method 2, K-dimensional) ---
def shape_discovery(S1, alpha, M):
    """
    2D strip idea extended to K dimensions.

    In 2D: for each y-strip we found the x-quantile (and vice versa).
    In K-D: for each dimension j (the "conditioning" axis) and each strip along j, it will find the (1-alpha) quantile along every
            other dimension i ≠ j.

    We will get a K×K structure:
        limits[j, i] is an array of M quantile thresholds where:
            - j is the dimension we bin along (the strip axis)
            - i is the dimension we compute the quantile for
        bin_edges[j] are the M+1 bin boundaries along dimension j

    A point s is inside the envelope if, for every dimension j, the value s[i] ≤ limits[j,i][bin(s[j])] for all i ≠ j.
    """

    N1, K = S1.shape

    # for each dimension, we ware creating M equal-width bins over [0, 1]
    bin_edges = np.zeros((K, M + 1)) 
    for j in range(K):
        bin_edges[j] = np.linspace(S1[:, j].min(), S1[:, j].max(), M + 1) 

    # limits[j, i, m] = (1-alpha) quantile of S1[:, i] for points whodse dim-j value falls in bin m
    limits = np.zeros((K, K, M))

    for j in range(K):                          # conditioning dimension
        for m in range(M):                      # index of the strip aalong dimension j
            lo, hi = bin_edges[j, m], bin_edges[j, m + 1]      # strip boundaries along dimension j
            in_strip = (S1[:, j] >= lo) & (S1[:, j] < hi)      # mask for points in the strip (boolean)

            for i in range(K):                  # dimensions to compute quantiles for
                if i == j:         
                    continue
                if in_strip.sum() < 3:    # 
                    # If there are very few points in the strip we use global quantile
                    limits[j, i, m] = np.quantile(S1[:, i], 1 - alpha)
                else:
                    limits[j, i, m] = np.quantile(S1[in_strip, i], 1 - alpha)

    # We need to ensure that the limits are non-increasing as the strip index increases 
    # (higher values of j shouldn't allow more of i).
    # Without this, the envelope can have holes.
    for j in range(K):
        for i in range(K):
            if i == j:
                continue
            for m in range(M - 2, -1, -1):
                limits[j, i, m] = max(limits[j, i, m], limits[j, i, m + 1])    # ensuring non-increasing llimits

    return bin_edges, limits


# --- Get Bin Indices ---
def get_bin_indices(scores, bin_edges):
    """
    For each point and each dimension, it finds which bin it falls in.
    """

    N, K = scores.shape
    M = bin_edges.shape[1] - 1    # number of bins is one less than the number of edges
    indices = np.zeros((N, K), dtype=int)    
    for j in range(K):
        raw = np.digitize(scores[:, j], bin_edges[j]) - 1   # digitize returns 1-based indices so converting it to 0-based
        indices[:, j] = np.clip(raw, 0, M - 1)      # 
    return indices

# --- Size Scaling (K-dimensional) ---
def size_scaling(S2, bin_edges, limits, alpha):
    """
    Finds t_hat by computing a tau score for each S2 point.

    tau for point s = max over all (j, i≠j) pairs of: s[i] / limits[j, i, bin_j(s)]

    that is how much would we need to scale all the limits to include s - we take the max ratio along all pairs of dimensions and comditions
    t_hat is then the (1-alpha) quantile of these tau scores.
    """
    N2, K = S2.shape
    bin_idx = get_bin_indices(S2, bin_edges)    # finding the bin

    # tau[n] = max over j, over i≠j of S2[n,i] / limits[j, i, bin_idx[n,j]]
    tau_scores = np.zeros(N2)
    for n in range(N2):
        max_ratio = 0.0
        for j in range(K):
            bj = bin_idx[n, j]
            for i in range(K):
                if i == j:
                    continue
                lim = limits[j, i, bj]
                if lim > 1e-12:
                    max_ratio = max(max_ratio, S2[n, i] / lim)
        tau_scores[n] = max_ratio

    n2 = N2
    idx = int(np.ceil((n2 + 1) * (1 - alpha))) - 1
    t_hat = np.sort(tau_scores)[np.clip(idx, 0, n2 - 1)]
    return t_hat, tau_scores


def size_scaling_fast(S2, bin_edges, limits, alpha):
    """
    Vectorised version of size_scaling — much faster for large N2.
    Avoids the loop over points.
    """

    N2, K = S2.shape
    M = bin_edges.shape[1] - 1
    bin_idx = get_bin_indices(S2, bin_edges)    

    # Building a (N2, K, K) array of ratios: ratio[n, j, i] = S2[n,i] / limits[j,i,bin_idx[n,j]]
    # limits[j, i, bin_idx[:, j]] has shape (N2,) for each (j,i) pair
    ratios = np.zeros((N2, K, K))
    for j in range(K):
        for i in range(K):
            if i == j:
                continue
            lims = limits[j, i, bin_idx[:, j]]  # (N2,) — limit for each point's bin
            ratios[:, j, i] = S2[:, i] / (lims + 1e-12)

    # tau = max over all (j, i≠j) pairs
    # Setting the diagonal to 0 (same-axis pairs are meaningless)
    tau_scores = ratios.max(axis=(1, 2))        

    n2 = N2
    idx = int(np.ceil((n2 + 1) * (1 - alpha))) - 1
    t_hat = np.sort(tau_scores)[np.clip(idx, 0, n2 - 1)]
    return t_hat, tau_scores


# --- Prediction (K-dimensional) ---
def is_in_region(scores, bin_edges, limits, t_hat):
    """
    A point s is inside the scaled region if for every pair (j, i≠j):
        s[i] ≤ t_hat * limits[j, i, bin_j(s)]
    """

    N, K = scores.shape
    bin_idx = get_bin_indices(scores, bin_edges)  
    scaled_limits = limits * t_hat                # scaling all the limits by t_hat

    inside = np.ones(N, dtype=bool)     # starting with all points inside, then marking those that violate any condition as false
    for j in range(K):
        for i in range(K):
            if i == j:  
                continue
            lims = scaled_limits[j, i, bin_idx[:, j]]   #limit for the bin for each point
            inside &= (scores[:, i] <= lims)      

    return inside

# --- Coverage Evaluation ---
def evaluate_coverage(S2, bin_edges, limits, t_hat):
    mask = is_in_region(S2, bin_edges, limits, t_hat)
    return mask.mean()

# ============================================================
# Example: runs for any K
# ============================================================
np.random.seed(42)
alpha = 0.1
M = 20          # bins per dimension — keep modest, data per bin shrinks fast

for K in [2, 5, 10]:
    scores = simulate_scores(n_phages=2000, K=K, correlation=0.3)
    S1, S2 = split_data(scores, split_ratio=0.5)

    bin_edges, limits = shape_discovery(S1, alpha, M)
    t_hat, _ = size_scaling_fast(S2, bin_edges, limits, alpha)
    coverage = evaluate_coverage(S2, bin_edges, limits, t_hat)

    print(f"K={K:2d} | t_hat={t_hat:.3f} | empirical coverage={coverage:.3f} " f"(target={1-alpha:.2f})")