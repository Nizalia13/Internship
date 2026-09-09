from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conformal_envelopes import (
    ConformalSetModel,
    evaluate_prediction_sets,
)


def toy_data(n=100, seed=4):
    rng = np.random.default_rng(seed)

    labels = np.array(
        ["A"] * (n // 2)
        + ["B"] * (n - n // 2)
    )

    p1 = np.r_[
        rng.normal(0.80, 0.05, n // 2),
        rng.normal(0.30, 0.05, n - n // 2),
    ]

    p2 = np.r_[
        rng.normal(0.75, 0.05, n // 2),
        rng.normal(0.35, 0.05, n - n // 2),
    ]

    p1 = np.clip(p1, 0, 1)
    p2 = np.clip(p2, 0, 1)

    p1[::17] = np.nan
    p2[::19] = np.nan

    return pd.DataFrame({
        "ID": [f"id{i}" for i in range(n)],
        "p1": p1,
        "p2": p2,
        "label": labels,
    })


@pytest.mark.parametrize(
    "method,params",
    [
        ("collapsed", {}),
        (
            "radial",
            {
                "n_directions": 50,
                "smoothing": 8.0,
                "angle_deg": 30.0,
            },
        ),
        (
            "strip",
            {
                "n_bins": 5,
                "smoothing_window": 2,
                "min_samples": 3,
            },
        ),
    ],
)
def test_fit_predict_all_methods(method, params):
    df = toy_data()

    model = ConformalSetModel(
        method=method,
        alpha=0.10,
        random_state=3,
        **params,
    )

    model.fit(df)
    result = model.predict(df)

    assert len(result) == len(df)
    assert set(model.get_envelopes()) == {"A", "B"}
    assert all(
        isinstance(value, list)
        for value in result["prediction_set"]
    )


def test_dict_output():
    df = toy_data()

    model = ConformalSetModel(
        method="collapsed",
    ).fit(df)

    result = model.predict(
        df.head(4),
        output="dict",
    )

    assert set(result) == set(
        df.head(4)["ID"]
    )


def test_missingness_mask():
    df = toy_data()

    model = ConformalSetModel(
        method="collapsed",
    ).fit(df)

    mask = model.missingness_mask(df)

    assert set(mask.columns) == {"p1", "p2"}
    assert set(np.unique(mask.to_numpy())) <= {0, 1}


def test_save_load(tmp_path):
    df = toy_data()

    model = ConformalSetModel(
        method="collapsed",
    ).fit(df)

    path = tmp_path / "model.pkl"
    model.save(path)

    loaded = ConformalSetModel.load(path)

    first = model.predict(df.head(5), output="dict")
    second = loaded.predict(df.head(5), output="dict")

    assert first == second


def test_metrics():
    predictions = [["A"], ["A", "B"], ["B"]]
    labels = ["A", "B", "A"]

    metrics = evaluate_prediction_sets(
        predictions,
        labels,
    )

    assert metrics["coverage"] == pytest.approx(2 / 3)
    assert metrics["average_set_size"] == pytest.approx(4 / 3)


def test_higher_is_better_requires_unit_interval():
    df = toy_data()
    df.loc[0, "p1"] = 2.0

    model = ConformalSetModel(
        method="collapsed",
    )

    with pytest.raises(ValueError):
        model.fit(df)
