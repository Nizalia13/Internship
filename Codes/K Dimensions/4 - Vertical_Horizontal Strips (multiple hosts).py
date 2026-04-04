import numpy as np

def simulate_scores(n, K, n_informative=2, variance=0.015, match_mean=None):
    """
    simulates scores for infectors and non-infectors
       - n: number of samples for each class
       - K: number of predictors per host
       - n_informative: number of predictors that have diffrent means
       - match_mean: mean vector for infectors """
    if match_mean is None:
        match_mean = np.full(K, 0.5)     # mean for non infectors
        match_mean[:n_informative] = 0.2    # mean for infectors

    nonmatch_mean = np.full(K, 0.5)  
    nonmatch_mean[:n_informative] = 0.7

    # simulating scores
    infectors = np.clip(np.random.multivariate_normal(match_mean, np.eye(K) * variance, n), 0, 1)
    # non- infectors are more spread out
    non_infectors = np.clip(np.random.multivariate_normal(nonmatch_mean, np.eye(K) * variance * 1.5, n), 0, 1)
    return infectors, non_infectors


#############3 Splitting the data ######################
def split_data(scores, split_ratio=0.5):
    """ 
    Splitting data randomly into two sets 
    """
    n   = len(scores)
    idx = np.random.permutation(n)
    cut = int(n * split_ratio)
    return scores[idx[:cut]], scores[idx[cut:]]


################### Shape Discovery #####################
def shape_discovery(S1, alpha, M):
    """
    For each conditioning dimension j and each of M bins along j,
    compute the (1-alpha) quantile of every other dimension i.
    Returns bin_edges (K, M+1) and limits (K, K, M).
    """
    N1, K = S1.shape      # number of samples and dimensions
    bin_edges = np.zeros((K, M + 1))           # bin edges for each dimension
    for j in range(K):                             
        bin_edges[j] = np.linspace(0, S1[:, j].max(), M + 1)   

    limits = np.zeros((K, K, M))     # limits[j, i, m] is the limit for dimension i when conditioning on dimension j in bin m
    for j in range(K):
        for m in range(M):
            lo, hi   = bin_edges[j, m], bin_edges[j, m + 1]
            in_strip = (S1[:, j] >= lo) & (S1[:, j] < hi)
            for i in range(K):
                if i == j: continue
                if in_strip.sum() < 3:
                    limits[j, i, m] = np.quantile(S1[:, i], 1 - alpha)
                else:
                    limits[j, i, m] = np.quantile(S1[in_strip, i], 1 - alpha)

    for j in range(K):
        for i in range(K):
            if i == j: continue
            for m in range(M - 2, -1, -1):
                limits[j, i, m] = max(limits[j, i, m], limits[j, i, m + 1])
    return bin_edges, limits


################# Size Scaling ###############################

def get_bin_indices(scores, bin_edges):
    N, K = scores.shape
    M    = bin_edges.shape[1] - 1
    indices = np.zeros((N, K), dtype=int)
    for j in range(K):
        raw = np.digitize(scores[:, j], bin_edges[j]) - 1
        indices[:, j] = np.clip(raw, 0, M - 1)
    return indices

def size_scaling(S2, bin_edges, limits, alpha):
    """
    tau(s) = max over all (j, i≠j) of s[i] / limits[j, i, bin_j(s)]
    t_hat  = (1-alpha) quantile of tau scores over S2
    """
    N2, K   = S2.shape
    bin_idx = get_bin_indices(S2, bin_edges)
    ratios = np.zeros((N2, K, K))
    for j in range(K):
        for i in range(K):
            if i == j: continue
            lims = limits[j, i, bin_idx[:, j]]
            ratios[:, j, i] = S2[:, i] / (lims + 1e-12)
    tau_scores = ratios.max(axis=(1, 2))
    idx   = int(np.ceil((N2 + 1) * (1 - alpha))) - 1
    t_hat = np.sort(tau_scores)[np.clip(idx, 0, N2 - 1)]
    return t_hat


################# per host envelope ####################
def is_in_region(scores, bin_edges, limits, t_hat):
    """
    A point s is inside if for each target dimension i,
    the tightest limit across all conditioning dims j holds.
    i.e. s[i] <= min_j( t_hat * limits[j, i, bin_j(s)] )
    """

    N, K          = scores.shape
    bin_idx       = get_bin_indices(scores, bin_edges)
    scaled_limits = limits * t_hat
    inside = np.ones(N, dtype=bool)
    for i in range(K):
        best_limit = np.full(N, np.inf)
        for j in range(K):
            if i == j: continue
            lims       = scaled_limits[j, i, bin_idx[:, j]]
            best_limit = np.minimum(best_limit, lims)
        inside &= (scores[:, i] <= best_limit)
    return inside


def build_envelope(infection_scores, alpha, M=20):
    """
    Builds a strip-based envelope from phages known to infect this host.
    Uses the same two-stage split and centring as the radial version.
    """

    S1, S2  = split_data(infection_scores)
    bin_edges, limits = shape_discovery(S1, alpha, M)
    t_hat = size_scaling(S2, bin_edges, limits, alpha)
    return {"bin_edges": bin_edges, "limits": limits, "t_hat": t_hat}



#################### Prediction ######################

INFECTS       = "infects"
NOT_INFECTS   = "does not infect"
UNCERTAIN     = "uncertain"
NO_PREDICTION = "NaN — both outside"


def predict_one_host(score_1, envelope):
    score_0 = 1.0 - score_1
    # Check if s1 (infection score) is in the envelope
    s1_in = bool(is_in_region(score_1[None, :], envelope["bin_edges"], 
                              envelope["limits"], envelope["t_hat"])[0])
    # Check if s0 (1 - s1) is in the envelope
    s0_in = bool(is_in_region(score_0[None, :], envelope["bin_edges"], 
                              envelope["limits"], envelope["t_hat"])[0])
    
    if s1_in and not s0_in:
        outcome = INFECTS
    elif s0_in and not s1_in: 
        outcome = NOT_INFECTS
    elif s1_in and s0_in:
         outcome = UNCERTAIN
    else:
        outcome = NO_PREDICTION
    
    return outcome, s1_in, s0_in


################## Execution ######################

hosts     = [f"Host_{chr(65+i)}" for i in range(6)]
true_host = "Host_B"
alpha     = 0.1
K         = 5
M         = 10

# Build Envelopes for all hosts (using simulated historical data)
envelopes = {}
for host in hosts:
    # Simulate historical 'infector' scores for this host
    mean_val = 0.2 if host == true_host else 0.5
    match_mean = np.full(K, mean_val)
    infectors = np.clip(np.random.normal(match_mean, 0.1, (500, K)), 0, 1)
    envelopes[host] = build_envelope(infectors, alpha, M=M)


# Define the scores for ONE single phage we want to test
# We simulate a specific phage here to show different outcomes
one_phage_scores = {}
for host in hosts:
    if host == "Host_A":
        one_phage_scores[host] = np.full(K, 0.5)   # Will likely be UNCERTAIN
    elif host == "Host_B":
        one_phage_scores[host] = np.full(K, 0.15)  # Will likely be INFECTS
    elif host == "Host_C":
        one_phage_scores[host] = np.full(K, 0.98)  # Will likely be NaN
    else:
        one_phage_scores[host] = np.full(K, 0.85)  # Will likely be DOES NOT INFECT


# Process the Phage against all Hosts
prediction_set = []
results = {}

for host in hosts:
    outcome, s1_in, s0_in = predict_one_host(one_phage_scores[host], envelopes[host])
    results[host] = {"outcome": outcome, "s1_in": s1_in, "s0_in": s0_in}
    
    # CRITICAL: Include both "infects" and "uncertain" in the prediction set
    if outcome in ("infects", "uncertain"):
        prediction_set.append(host)


# Final Output
print("=" * 65)
print(f"SINGLE PHAGE ANALYSIS | alpha={alpha}")
print("=" * 65)
print(f"{'Target Host':<15} | {'s1_in':<8} | {'s0_in':<8} | {'Outcome'}")
print("-" * 65)

for host in hosts:
    r = results[host]
    s1_txt = "YES" if r['s1_in'] else "NO"
    s0_txt = "YES" if r['s0_in'] else "NO"
    print(f"{host:<15} | {s1_txt:<8} | {s0_txt:<8} | {r['outcome']}")

print("-" * 65)
print(f"FINAL PREDICTION SET: {prediction_set}")