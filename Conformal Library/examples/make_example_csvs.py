from pathlib import Path

import numpy as np
import pandas as pd


rng = np.random.default_rng(10)


def make_dataset(n, start=0):
    labels = rng.choice(["A", "B"], size=n)

    X = np.zeros((n, 3))

    for i, label in enumerate(labels):
        if label == "A":
            center = np.array([0.88, 0.78, 0.83])
        else:
            center = np.array([0.38, 0.30, 0.42])

        X[i] = np.clip(
            center + rng.normal(0, 0.08, 3),
            0,
            1,
        )

    # Structural missingness demonstration.
    X[rng.random(X.shape) < 0.08] = np.nan

    return pd.DataFrame({
        "ID": [f"id_{start+i:04d}" for i in range(n)],
        "score_1": X[:, 0],
        "score_2": X[:, 1],
        "score_3": X[:, 2],
        "label": labels,
    })


here = Path(__file__).resolve().parent

train = make_dataset(200)
test = make_dataset(60, start=200)

train.to_csv(here / "train_example.csv", index=False)
test.to_csv(here / "test_example.csv", index=False)

print("Wrote train_example.csv and test_example.csv")
