import numpy as np

np.random.seed(42)

################# Data Simulation ##################
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



################### Data Splitting ######################
def split_data(scores, split_ratio=0.5):
    """ 
    Splitting data randomly into two sets 
    """
    n   = len(scores)
    idx = np.random.permutation(n)
    cut = int(n * split_ratio)
    return scores[idx[:cut]], scores[idx[cut:]]


################## Shape Discovery ########################
def sample_positive_sphere(M, K):
    """ 
    Samples M points uniformly on the positive orthant of the unit sphere in K dimensions
    """
    V = np.abs(np.random.randn(M, K))
    return V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)


def shape_discovery(S1, alpha, M, delta_deg=10):
    """
    Finds M projection direction and their corresponding quantiles to build the envelope
    For each direction, we only consider the scores which are within delta_deg of that direction
    """

    K = S1.shape[1]    # number of score dimensions
    U = sample_positive_sphere(M, K)    # M random directions on K dimensions (positive)
    mags = np.linalg.norm(S1, axis=1)    # magnitudes of the score vectors
    dirs = S1 / (mags[:, None] + 1e-12)    # threshold for selecting scores
    cos_thresh = np.cos(np.radians(delta_deg))    
    q_tilde    = np.zeros(M)  
    for m in range(M):    
        sims = dirs @ U[m]      # closeness of each score direction to the m-th random direction
        mask = sims >= cos_thresh   # selecting scores that are close enough
        if mask.sum() < 5:
            q_tilde[m] = np.quantile(mags, 1 - alpha)     # if not enough scores are close enough we use all scores to compute the qunatile
        else:
            q_tilde[m] = np.quantile(mags[mask], 1 - alpha)
    return U, q_tilde



################## Size Scaling ########################
def size_scaling(S2, U, q_tilde, alpha):
    """
    Finds t_hat so the envelope defined by U and q_tilde has 1-alpha coverage on S2
    tau_scores = min over m of [ max over k of s[k] / boundary[m,k] ]
    t_hat  = (1-alpha) quantile of tau scores over S2]
    """
    boundary = U * q_tilde[:, None]      # the envelope boundary in each direction
    ratios = S2[:, None, :] / (boundary[None, :, :] + 1e-12)      # how much each score exceeds the boundary in each direction
    t_per_sector = ratios.max(axis=2)    # for each score and each sector  we take the max ratio
    tau_scores = t_per_sector.min(axis=1)   # for each score we take the min ratio across sectors
    n2 = len(tau_scores)     
    idx = int(np.ceil((n2 + 1) * (1 - alpha))) - 1     # index for 1-alpha quantile
    t_hat = np.sort(tau_scores)[np.clip(idx, 0, n2 - 1)]    # 1-alpha quantile of tau scores
    return t_hat, tau_scores



################## Constrcuting the Envelope ####################
def is_in_region(score_vector, U, q_tilde, t_hat):
    """
    Checking if a score_vector is inside the scaled envelope
    a point is inside if any of the sectors contains it (all components of scores <= scaled boundary for that sector)
    points closer to the origin are alwaya inside
    """
    q_final  = q_tilde * t_hat      # final quantiles after scaling
    boundary = U * q_final[:, None]       # envelope boundary in each direction
    return np.any(np.all(score_vector[:, None, :] <= boundary[None, :, :], axis=2),axis=1)


def build_envelope(infection_scores, alpha, M=200, delta_deg=15):
    """ 
    Builds an envelope for  the given host
    """
    S1, S2     = split_data(infection_scores)     
    U, q_tilde = shape_discovery(S1, alpha, M, delta_deg)   
    t_hat, _   = size_scaling(S2, U, q_tilde, alpha)
    return {"U": U, "q_tilde": q_tilde, "t_hat": t_hat}  


def check_score(score_vector, envelope):
    """
    Checks if a given score vector is inside the envelope
    Returns true is a point is inside, false otherwise
    """
    return bool(is_in_region(score_vector[None, :], envelope["U"], envelope["q_tilde"], envelope["t_hat"])[0])


################# Prediction ######################
INFECTS = "infects"
NOT_INFECTS = "does not infect"
UNCERTAIN = "uncertain"
NO_PREDICTION = "NaN — both outside"


def predict_one_host(score_1, envelope):
    """
    Predicts if a test score indicates infection, non infection or is uncertain
    """
    score_0 = 1.0 - score_1     
    s1_in = check_score(score_1, envelope)   
    s0_in = check_score(score_0, envelope)
    if s1_in and not s0_in: 
        outcome = INFECTS
    elif s0_in and not s1_in: 
        outcome = NOT_INFECTS
    elif s1_in and s0_in:     
        outcome = UNCERTAIN
    else:                     
        outcome = NO_PREDICTION
    return outcome, s1_in, s0_in


def predict_phage_hosts(score_1_per_host, envelopes, hosts):
    prediction_set = []
    results = {}
    for host in hosts:
        outcome, s1_in, s0_in = predict_one_host(score_1_per_host[host], envelopes[host])  # for each host we check if the score is inside or not
        if outcome in (INFECTS, UNCERTAIN):
            prediction_set.append(host)      # if the score is inside the envelope we predict infection or uncertainity
        results[host] = {"outcome": outcome, "s1_in": s1_in, "s0_in": s0_in}     # we also store the results for each host
    return prediction_set, results


    #################### Execution #########################
# def evaluate_coverage(hosts, true_host, envelopes, calibration_data, alpha, N_test=300):
#     """
#     Evaluates the coverage of the envelopes by simulating test scores 
#     for the true host and other hosts.
#     """
#     found_true     = 0
#     pred_sizes     = []
#     # Initialize counts for all possible outcomes
#     outcome_counts = {INFECTS: 0, NOT_INFECTS: 0, UNCERTAIN: 0, NO_PREDICTION: 0}
    
#     for _ in range(N_test):
#         score_1_per_host = {}
#         for host in hosts:
#             infectors, non_infectors = calibration_data[host]   
#             mc, std = infectors.mean(axis=0), infectors.std(axis=0)
#             nmc, nstd = non_infectors.mean(axis=0), non_infectors.std(axis=0)  
#             K = mc.shape[0]     
            
#             if host == true_host:   
#                 score_1_per_host[host] = np.clip(mc + np.random.randn(K) * std, 0, 1) 
#             else:
#                 score_1_per_host[host] = np.clip(nmc + np.random.randn(K) * nstd * 1.2, 0, 1)
        
#         # Now score_1_per_host exists and can be passed to the predictor
#         pred_set, results = predict_phage_hosts(score_1_per_host, envelopes, hosts)
#         pred_sizes.append(len(pred_set))
        
#         # 1. Track Coverage: Was the true host in the prediction set?
#         if true_host in pred_set:
#             found_true += 1
            
#         # 2. Track Outcome Distribution: What happened to the true host?
#         # We look at 'results[true_host]' regardless of whether it made it into 'pred_set'
#         actual_outcome = results[true_host]["outcome"]
#         outcome_counts[actual_outcome] += 1
        
#     return found_true / N_test, np.mean(pred_sizes), outcome_counts


# hosts         = [f"Host_{chr(65+i)}" for i in range(6)]
# true_host     = "Host_B"
# alpha         = 0.1
# K             = 5
# n_informative = 2
# N_cal         = 1000
# N_test        = 500
# M             = 200
# delta_deg     = 15

# np.random.seed(42)

# calibration_data = {}
# envelopes        = {}

# for host in hosts:
#     match_mean = np.random.uniform(0.1, 0.6, K)
#     match_mean[:n_informative] = np.random.uniform(0.1, 0.35, n_informative)
#     # Note: ensure simulate_scores is defined in your script
#     infectors, non_infectors = simulate_scores(
#         N_cal, K, n_informative=n_informative, match_mean=match_mean)
#     calibration_data[host] = (infectors, non_infectors)
#     envelopes[host]        = build_envelope(
#         infectors, alpha, M=M, delta_deg=delta_deg)

# score_1_demo = {}
# for host in hosts:
#     infectors, non_infectors = calibration_data[host]
#     mc, std   = infectors.mean(axis=0),     infectors.std(axis=0)
#     nmc, nstd = non_infectors.mean(axis=0), non_infectors.std(axis=0)
#     if host == true_host:
#         score_1_demo[host] = np.clip(mc + np.random.randn(K) * std, 0, 1)
#     else:
#         score_1_demo[host] = np.clip(nmc + np.random.randn(K) * nstd * 1.2, 0, 1)

# pred_set, results = predict_phage_hosts(score_1_demo, envelopes, hosts)

# print("=" * 55)
# print(f"METHOD 1 — RADIAL  |  alpha={alpha}  |  true host: {true_host}")
# print("=" * 55)
# print(f"  {'Host':<12} {'s1_in':>6} {'s0_in':>6}  outcome")
# print("  " + "-" * 45)
# for host in hosts:
#     r      = results[host]
#     marker = "  <-- TRUE" if host == true_host else ""
#     print(f"  {host:<12} {'Y' if r['s1_in'] else 'N':>6}"
#           f" {'Y' if r['s0_in'] else 'N':>6}  {r['outcome']}{marker}")
# print(f"\n  Prediction set: {pred_set}")

# coverage, avg_size, outcome_counts = evaluate_coverage(
#     hosts, true_host, envelopes, calibration_data, alpha, N_test)

# print(f"\n--- Coverage over {N_test} test phages ---")
# print(f"  Empirical coverage : {coverage:.3f}  (target >= {1-alpha:.2f})")
# print(f"  Avg prediction set : {avg_size:.2f} hosts  (out of {len(hosts)})")
# print(f"  True host outcomes : "
#       f"infects={outcome_counts[INFECTS]}, "
#       f"uncertain={outcome_counts[UNCERTAIN]}, "
#       f"not_infects={outcome_counts[NOT_INFECTS]}, "
#       f"NaN={outcome_counts[NO_PREDICTION]}")



hosts         = [f"Host_{chr(65+i)}" for i in range(6)]
true_host     = "Host_B"  # The host this specific phage actually infects
alpha         = 0.1
K             = 5
N_cal         = 1000
M             = 200
delta_deg     = 15

calibration_data = {}
envelopes        = {}

for host in hosts:
    # Simulate historical infection data for each host to build its "profile"
    match_mean = np.random.uniform(0.1, 0.6, K)
    infectors, _ = simulate_scores(N_cal, K, match_mean=match_mean)
    
    calibration_data[host] = infectors
    envelopes[host] = build_envelope(infectors, alpha, M=M, delta_deg=delta_deg)


# Simulate scores for one phage. 
single_phage_scores = {}
for host in hosts:
    if host == true_host:
        # Generate a score that looks like a successful infection for Host B
        # We use the mean of Host B's known infectors
        mean_score = calibration_data[host].mean(axis=0)
        single_phage_scores[host] = np.clip(mean_score + np.random.randn(K) * 0.05, 0, 1)
    else:
        # Generate a "non-infecting" score (e.g., center around 0.7 for non-match)
        single_phage_scores[host] = np.random.uniform(0.6, 0.9, K)


# This checks s1 (score) and s0 (1 - score) against every host envelope
pred_set, results = predict_phage_hosts(single_phage_scores, envelopes, hosts)

# 5. OUTPUT THE RESULTS
print("=" * 60)
print(f"SINGLE PHAGE TEST RESULTS (alpha={alpha})")
print("=" * 60)
print(f"{'Target Host':<15} | {'s1_in':<8} | {'s0_in':<8} | {'Outcome'}")
print("-" * 60)

for host in hosts:
    res = results[host]
    s1_text = "YES" if res['s1_in'] else "NO"
    s0_text = "YES" if res['s0_in'] else "NO"
    marker = " <-- (Actual Host)" if host == true_host else ""
    
    print(f"{host:<15} | {s1_text:<8} | {s0_text:<8} | {res['outcome']}{marker}")

print("-" * 60)
print(f"FINAL PREDICTION SET: {pred_set}")



### Second phage

single_phage_scores = {}

for host in hosts:
    if host == "Host_A":
        single_phage_scores[host] = np.full(K, 0.5)
        
    elif host == true_host: # Host_B
        mc = calibration_data[host].mean(axis=0)
        single_phage_scores[host] = np.clip(mc + np.random.randn(K) * 0.02, 0, 1)
        
    elif host == "Host_C":
        single_phage_scores[host] = np.full(K, 0.99)
        
    else:
        single_phage_scores[host] = np.full(K, 0.8)

pred_set, results = predict_phage_hosts(single_phage_scores, envelopes, hosts)


print("=" * 65)
print(f"SINGLE PHAGE TEST | alpha={alpha} | true host: {true_host}")
print("=" * 65)
print(f"  {'Host':<12} | {'s1_in':<6} | {'s0_in':<6} | {'Outcome'}")
print("  " + "-" * 55)

for host in hosts:
    r = results[host]
    s1_txt = "YES" if r['s1_in'] else "NO"
    s0_txt = "YES" if r['s0_in'] else "NO"
    marker = "  <-- TRUE" if host == true_host else ""
    
    print(f"  {host:<12} | {s1_txt:<6} | {s0_txt:<6} | {r['outcome']}{marker}")

print(f"\nFINAL PREDICTION SET: {pred_set}")


