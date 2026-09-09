from __future__ import annotations

import numpy as np


def evaluate_prediction_sets(prediction_sets, true_labels):
    """Evaluate set-valued predictions."""
    prediction_sets = list(prediction_sets)
    true_labels = list(true_labels)

    if len(prediction_sets) != len(true_labels):
        raise ValueError(
            "prediction_sets and true_labels must have the same length."
        )

    if len(true_labels) == 0:
        raise ValueError("Cannot evaluate an empty dataset.")

    covered = [
        true_labels[i] in prediction_sets[i]
        for i in range(len(true_labels))
    ]
    sizes = [len(s) for s in prediction_sets]
    singleton_idx = [
        i for i, s in enumerate(prediction_sets)
        if len(s) == 1
    ]

    return {
        "coverage": float(np.mean(covered)),
        "average_set_size": float(np.mean(sizes)),
        "singleton_rate": float(
            len(singleton_idx) / len(true_labels)
        ),
        "singleton_accuracy": (
            float(np.mean([
                prediction_sets[i][0] == true_labels[i]
                for i in singleton_idx
            ]))
            if singleton_idx else None
        ),
        "empty_rate": float(np.mean([
            len(s) == 0 for s in prediction_sets
        ])),
    }
