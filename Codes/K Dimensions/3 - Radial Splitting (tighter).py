import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

#############################################################
#  DATA SIMULATION  (K-dimensional, cluster anywhere)
#############################################################

def simulate_host_data(n, K, match_mean=None, nonmatch_mean=None, variance=0.02):
    """
    Match and non-match clusters can live anywhere in [0,1]^K.
    Defaults place matches mid-range and non-matches spread out,
    but real data can have any distribution.
    """
    if match_mean    is None: match_mean    = np.full(K, 0.3)
    if nonmatch_mean is None: nonmatch_mean = np.full(K, 0.7)

    matches     = np.clip(np.abs(np.random.multivariate_normal(
                    match_mean,    np.eye(K) * variance,       n)), 0, 1)
    non_matches = np.clip(np.abs(np.random.multivariate_normal(
                    nonmatch_mean, np.eye(K) * variance * 1.5, n)), 0, 1)
    return matches, non_matches


#############################################################
#  SHARED UTILITIES
#############################################################

def sample_positive_sphere(M, K):
    """Uniform directions on S^(K-1)_+."""
    V = np.abs(np.random.randn(M, K))
    return V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)


def split_calibration_data(scores, split_ratio=0.5):
    n   = len(scores)
    idx = np.random.permutation(n)
    cut = int(n * split_ratio)
    return scores[idx[:cut]], scores[idx[cut:]]


#############################################################
#  METHOD 1 — RADIAL / ANGULAR  (non-convex star-shaped)
#############################################################

def m1_fit(S1, alpha, M, delta_deg=15):
    """
    Shape discovery for Method 1.
    For each direction u_m, find the (1-alpha) quantile of magnitudes
    of S1 points whose direction is within delta_deg of u_m.

    Note: scores are centred before fitting so the envelope wraps
    tightly around wherever the cluster actually lives.
    """
    centre = S1.mean(axis=0)
    S1c    = S1 - centre                             # centre at origin

    K  = S1.shape[1]
    U  = sample_positive_sphere(M, K)

    # Work in the FULL sphere (not just positive orthant) after centring,
    # since centred scores can be negative.
    # So we re-sample U from the full sphere here.
    V  = np.random.randn(M, K)
    U  = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)

    mags = np.linalg.norm(S1c, axis=1)
    dirs = S1c / (mags[:, None] + 1e-12)

    cos_thresh = np.cos(np.radians(delta_deg))
    q_tilde    = np.zeros(M)

    for m in range(M):
        sims = dirs @ U[m]
        mask = sims >= cos_thresh
        if mask.sum() < 5:
            q_tilde[m] = np.quantile(mags, 1 - alpha)
        else:
            q_tilde[m] = np.quantile(mags[mask], 1 - alpha)

    # Size scaling: find t_hat on the second half of S1
    # (we split S1 internally here so the caller doesn't have to)
    S1a, S1b   = split_calibration_data(S1c, split_ratio=0.5)

    # Refit on S1a
    mags_a = np.linalg.norm(S1a, axis=1)
    dirs_a = S1a / (mags_a[:, None] + 1e-12)
    for m in range(M):
        sims = dirs_a @ U[m]
        mask = sims >= cos_thresh
        if mask.sum() < 5:
            q_tilde[m] = np.quantile(mags_a, 1 - alpha)
        else:
            q_tilde[m] = np.quantile(mags_a[mask], 1 - alpha)

    # Compute t_hat on S1b
    boundary     = U * q_tilde[:, None]              # (M, K)
    ratios       = S1b[:, None, :] / (boundary[None, :, :] + 1e-12)
    t_per_sector = ratios.max(axis=2)
    tau_scores   = t_per_sector.min(axis=1)
    n2           = len(tau_scores)
    idx          = int(np.ceil((n2 + 1) * (1 - alpha))) - 1
    t_hat        = np.sort(tau_scores)[np.clip(idx, 0, n2 - 1)]

    return {"method": "m1", "U": U, "q_tilde": q_tilde * t_hat, "centre": centre}


def m1_check(score_vector, envelope):
    """True if score_vector is inside the Method 1 envelope."""
    sc       = score_vector - envelope["centre"]     # centre the test point
    U        = envelope["U"]
    q_tilde  = envelope["q_tilde"]
    boundary = U * q_tilde[:, None]                  # (M, K)
    # Inside if ANY sector covers it (all K dims <= boundary for that sector)
    return bool(np.any(
        np.all(sc[None, :] <= boundary, axis=1)
    ))


#############################################################
#  METHOD 2 — STRIPS / STAIRCASE  (non-convex axis-aligned)
#############################################################

def m2_fit(S1, alpha, M=20):
    """
    Shape discovery for Method 2.
    Bins each dimension into M strips and computes conditional quantiles.
    Also centred so it handles any cluster location.
    """
    centre = S1.mean(axis=0)
    S1c    = S1 - centre

    N1, K  = S1c.shape

    # Bin edges for each dimension
    bin_edges = np.zeros((K, M + 1))
    for j in range(K):
        bin_edges[j] = np.linspace(S1c[:, j].min(), S1c[:, j].max(), M + 1)

    # limits[j, i, m] = (1-alpha) quantile of dim i in strip m of dim j
    limits = np.zeros((K, K, M))
    for j in range(K):
        for m in range(M):
            lo, hi    = bin_edges[j, m], bin_edges[j, m + 1]
            in_strip  = (S1c[:, j] >= lo) & (S1c[:, j] < hi)
            for i in range(K):
                if i == j: continue
                if in_strip.sum() < 3:
                    limits[j, i, m] = np.quantile(S1c[:, i], 1 - alpha)
                else:
                    limits[j, i, m] = np.quantile(S1c[in_strip, i], 1 - alpha)

    # Monotonicity fix
    for j in range(K):
        for i in range(K):
            if i == j: continue
            for m in range(M - 2, -1, -1):
                limits[j, i, m] = max(limits[j, i, m], limits[j, i, m + 1])

    # Size scaling: t_hat via vectorised tau scores
    S1a, S1b = split_calibration_data(S1c, split_ratio=0.5)

    # Refit limits on S1a only
    bin_edges_a = np.zeros((K, M + 1))
    for j in range(K):
        bin_edges_a[j] = np.linspace(S1a[:, j].min(), S1a[:, j].max(), M + 1)
    limits_a = np.zeros((K, K, M))
    for j in range(K):
        for m in range(M):
            lo, hi   = bin_edges_a[j, m], bin_edges_a[j, m + 1]
            in_strip = (S1a[:, j] >= lo) & (S1a[:, j] < hi)
            for i in range(K):
                if i == j: continue
                if in_strip.sum() < 3:
                    limits_a[j, i, m] = np.quantile(S1a[:, i], 1 - alpha)
                else:
                    limits_a[j, i, m] = np.quantile(S1a[in_strip, i], 1 - alpha)
    for j in range(K):
        for i in range(K):
            if i == j: continue
            for m in range(M - 2, -1, -1):
                limits_a[j, i, m] = max(limits_a[j, i, m], limits_a[j, i, m + 1])

    # Compute t_hat on S1b
    N2      = len(S1b)
    bin_idx = np.zeros((N2, K), dtype=int)
    for j in range(K):
        raw           = np.digitize(S1b[:, j], bin_edges_a[j]) - 1
        bin_idx[:, j] = np.clip(raw, 0, M - 1)

    ratios = np.zeros((N2, K, K))
    for j in range(K):
        for i in range(K):
            if i == j: continue
            lims              = limits_a[j, i, bin_idx[:, j]]
            ratios[:, j, i]   = S1b[:, i] / (lims + 1e-12)
    tau_scores = ratios.max(axis=(1, 2))
    n2         = len(tau_scores)
    idx        = int(np.ceil((n2 + 1) * (1 - alpha))) - 1
    t_hat      = np.sort(tau_scores)[np.clip(idx, 0, n2 - 1)]

    return {
        "method":     "m2",
        "bin_edges":  bin_edges,
        "limits":     limits * t_hat,
        "centre":     centre,
    }


def m2_check(score_vector, envelope):
    """True if score_vector is inside the Method 2 envelope."""
    sc        = score_vector - envelope["centre"]
    bin_edges = envelope["bin_edges"]
    limits    = envelope["limits"]
    K         = len(sc)
    M         = bin_edges.shape[1] - 1

    bin_idx = np.zeros(K, dtype=int)
    for j in range(K):
        raw        = np.digitize(sc[j], bin_edges[j]) - 1
        bin_idx[j] = np.clip(raw, 0, M - 1)

    for j in range(K):
        for i in range(K):
            if i == j: continue
            if sc[i] > limits[j, i, bin_idx[j]]:
                return False
    return True


#############################################################
#  UNIFIED INTERFACE
#############################################################

def get_host_envelope(training_matches, alpha, M, method="m1", delta_deg=15):
    """
    Single entry point — returns an envelope dict regardless of method.
    method: "m1" (radial) or "m2" (strips)
    """
    if method == "m1":
        return m1_fit(training_matches, alpha, M, delta_deg=delta_deg)
    else:
        return m2_fit(training_matches, alpha, M)


def check_envelope(score_vector, envelope):
    """Single entry point for checking a score against any envelope."""
    if envelope["method"] == "m1":
        return m1_check(score_vector, envelope)
    else:
        return m2_check(score_vector, envelope)


#############################################################
#  INFERENCE
#############################################################

def predict_hosts(test_s1, test_s0, envelopes, host_names):
    prediction_set = []
    results        = {}
    for host in host_names:
        s1_in = check_envelope(test_s1[host], envelopes[host])
        s0_in = check_envelope(test_s0[host], envelopes[host])

        if   s1_in and not s0_in: decision = "confident match"
        elif s0_in and not s1_in: decision = "confident non-match"
        elif s1_in and s0_in:     decision = "uncertain — kept"
        else:                     decision = "both outside — kept"

        if decision != "confident non-match":
            prediction_set.append(host)

        results[host] = {"s1_in": s1_in, "s0_in": s0_in, "decision": decision}
    return prediction_set, results


#############################################################
#  VISUALISATION  (2D only)
#############################################################

def plot_comparison_2d(hosts, calibration_data, env_m1, env_m2,
                       test_s1, test_s0, alpha, true_host):
    """Side-by-side Method 1 vs Method 2 for each host (K=2 only)."""
    n     = len(hosts)
    fig, axes = plt.subplots(n, 2, figsize=(11, 4.5 * n))
    if n == 1: axes = axes[None, :]

    xs, ys   = np.linspace(-0.5, 0.5, 120), np.linspace(-0.5, 0.5, 120)
    X, Y     = np.meshgrid(xs, ys)
    grid_raw = np.vstack([X.ravel(), Y.ravel()]).T  # centred grid

    for row, host in enumerate(hosts):
        matches, non_matches = calibration_data[host]

        for col, (env, label) in enumerate([
                (env_m1[host], "Method 1 — radial"),
                (env_m2[host], "Method 2 — strips")]):

            ax = axes[row, col]
            c  = env["centre"]

            # Plot calibration points (shift to centred space)
            ax.scatter(matches[:,0]-c[0],     matches[:,1]-c[1],
                       color='green', alpha=0.3, s=10, label="matches")
            ax.scatter(non_matches[:,0]-c[0], non_matches[:,1]-c[1],
                       color='red',   alpha=0.1, s=6,  label="non-matches")

            # Draw envelope in centred space
            inside = np.array([
                check_envelope(g + c, env) for g in grid_raw
            ]).reshape(X.shape)
            ax.contour( X, Y, inside, levels=[0.5], colors='royalblue', linewidths=2)
            ax.contourf(X, Y, inside, levels=[0.5, 1],
                        colors='royalblue', alpha=0.08)

            # Test points (also centred)
            s1c = test_s1[host] - c
            s0c = test_s0[host] - c
            s1_in = check_envelope(test_s1[host], env)
            s0_in = check_envelope(test_s0[host], env)

            ax.scatter(s1c[0], s1c[1], color='blue',  marker='P', s=160,
                       edgecolors='white', zorder=5,
                       label=f"s1 ({'IN' if s1_in else 'OUT'})")
            ax.scatter(s0c[0], s0c[1], color='black', marker='X', s=160,
                       edgecolors='white', zorder=5,
                       label=f"s0 ({'IN' if s0_in else 'OUT'})")

            true_marker = "  TRUE HOST" if host == true_host else ""
            ax.set_title(f"{host}{true_marker}  |  {label}", fontsize=10)
            ax.set_xlabel("dim 1 (centred)"); ax.set_ylabel("dim 2 (centred)")
            ax.legend(fontsize=7, loc='upper right')
            ax.grid(alpha=0.2)
            ax.set_xlim(-0.5, 0.5); ax.set_ylim(-0.5, 0.5)

    plt.suptitle(f"Non-convex envelopes per host  [{int((1-alpha)*100)}% coverage]",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/host_envelopes_nonconvex.png',
                dpi=130, bbox_inches='tight')
    plt.show()


def print_results(hosts, results_m1, results_m2, pred_m1, pred_m2, true_host, alpha):
    print(f"\nalpha={alpha}  |  true host: {true_host}\n")
    print(f"{'Host':<12} {'M1 s1':>7} {'M1 s0':>7} {'M1 decision':<26}"
          f"{'M2 s1':>7} {'M2 s0':>7} {'M2 decision'}")
    print("-" * 90)
    for host in hosts:
        r1, r2 = results_m1[host], results_m2[host]
        t      = " <-- TRUE" if host == true_host else ""
        print(f"  {host:<10}"
              f" {'Y' if r1['s1_in'] else 'N':>7}"
              f" {'Y' if r1['s0_in'] else 'N':>7}"
              f"  {r1['decision']:<24}"
              f" {'Y' if r2['s1_in'] else 'N':>7}"
              f" {'Y' if r2['s0_in'] else 'N':>7}"
              f"  {r2['decision']}{t}")
    print(f"\nM1 prediction set: {pred_m1}")
    print(f"M2 prediction set: {pred_m2}")


#############################################################
#  EXECUTION
#############################################################

if __name__ == "__main__":
    np.random.seed(42)

    hosts     = ['Host A', 'Host B', 'Host C', 'Host D']
    true_host = 'Host B'
    alpha     = 0.1
    M         = 150
    K         = 10      # set to any K; plots only appear for K=2

    # Calibration
    calibration_data = {}
    env_m1, env_m2   = {}, {}

    for host in hosts:
        # Vary the match cluster location per host to make it interesting
        match_mean = np.random.uniform(0.2, 0.6, K)
        matches, non_matches    = simulate_host_data(500, K,
                                    match_mean=match_mean, variance=0.015)
        calibration_data[host]  = (matches, non_matches)
        env_m1[host] = get_host_envelope(matches, alpha, M,
                                         method="m1", delta_deg=15)
        env_m2[host] = get_host_envelope(matches, alpha, M,
                                         method="m2")

    # Test scores — phage truly infects Host B
    test_s1, test_s0 = {}, {}
    for host in hosts:
        mc = calibration_data[host][0].mean(axis=0)  # match cluster centre
        if host == true_host:
            # s1 looks like a match: near the match cluster
            test_s1[host] = mc + np.random.uniform(-0.05, 0.05, K)
            # s0 looks wrong: far from the match cluster
            test_s0[host] = mc + np.random.uniform( 0.25, 0.40, K)
        else:
            # s1 looks wrong: far from this host's match cluster
            test_s1[host] = mc + np.random.uniform( 0.25, 0.40, K)
            # s0 looks right: near the match cluster (non-infection is typical)
            test_s0[host] = mc + np.random.uniform(-0.05, 0.05, K)

    # Inference
    pred_m1, results_m1 = predict_hosts(test_s1, test_s0, env_m1, hosts)
    pred_m2, results_m2 = predict_hosts(test_s1, test_s0, env_m2, hosts)

    print_results(hosts, results_m1, results_m2, pred_m1, pred_m2, true_host, alpha)

    if K == 2:
        plot_comparison_2d(hosts, calibration_data,
                           env_m1, env_m2,
                           test_s1, test_s0, alpha, true_host)