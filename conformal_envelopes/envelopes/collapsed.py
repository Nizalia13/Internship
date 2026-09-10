from __future__ import annotations

import numpy as np

from ..calibration import conformal_quantile
from ..missingness import nanmean_rows

EPS = 1e-12


def build_collapsed(S1, S2, alpha: float) -> dict:
    """Build the collapsed 1D envelope used in the cleaned notebook."""
    S1_1d = nanmean_rows(S1, all_missing_value=0.0)
    S2_1d = nanmean_rows(S2, all_missing_value=0.0)

    q_tilde = float(np.nanquantile(S1_1d, 1.0 - alpha))
    tau_scores = S2_1d / (q_tilde + EPS)
    t_hat = conformal_quantile(tau_scores, alpha)

    return {
        "method": "collapsed",
        "q_tilde": q_tilde,
        "t_hat": t_hat,
    }


def collapsed_tau(scores, envelope: dict) -> np.ndarray:
    scores_1d = nanmean_rows(scores, all_missing_value=0.0)
    return scores_1d / (
        float(envelope["q_tilde"]) * float(envelope["t_hat"]) + EPS
    )


def collapsed_is_in_region(scores, envelope: dict) -> np.ndarray:
    return collapsed_tau(scores, envelope) <= 1.0
