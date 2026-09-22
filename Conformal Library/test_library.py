import numpy as np
import pandas as pd
import pytest

from conformal_envelopes import (ConformalSetModel, evaluate_prediction_sets,)

###############################################################################################################

def toy_data(n=100, seed=4):
    rng = np.random.default_rng(seed)

    labels = np.array(["A"] * (n // 2) + ["B"] * (n - n // 2))

    p1 = np.r_[rng.normal(0.80, 0.05, n // 2), rng.normal(0.30, 0.05, n - n // 2),]

    p2 = np.r_[rng.normal(0.75, 0.05, n // 2), rng.normal(0.35, 0.05, n - n // 2),]

    p1 = np.clip(p1, 0, 1)
    p2 = np.clip(p2, 0, 1)

    p1[::17] = np.nan
    p2[1::19] = np.nan  

    return pd.DataFrame({"ID": [f"id{i}" for i in range(n)], "p1": p1, 
                         "p2": p2, "label": labels,})

###############################################################################################################

@pytest.mark.parametrize("method,params", [("collapsed", {}), 
                                           ("radial",{"n_directions": 50, "smoothing": 8.0,
                                                      "angle_deg": 30.0,},),
                                           ("strip", {"n_bins": 5, "min_samples": 3,},),],)

########

def test_fit_predict_all_methods(method, params):
    df = toy_data()

    model = ConformalSetModel(method=method, alpha=0.10, random_state=3, **params,)

    model.fit(df)
    result = model.predict(df)

    assert len(result) == len(df)
    assert set(model.get_envelopes()) == {"A", "B"}
    assert all(isinstance(value, list) for value in result["prediction_set"])

###############################################################################################################

def test_dict_output():
    df = toy_data()

    model = ConformalSetModel(method="collapsed",).fit(df)

    result = model.predict(df.head(4), output="dict",)

    assert set(result) == set(df.head(4)["ID"])

###############################################################################################################

def test_missingness_mask():
    df = toy_data()

    model = ConformalSetModel(method="collapsed",).fit(df)

    mask = model.missingness_mask(df)

    assert set(mask.columns) == {"p1", "p2"}
    assert set(np.unique(mask.to_numpy())) <= {0, 1}

###############################################################################################################

def test_metrics():
    predictions = [["A"], ["A", "B"], ["B"]]
    labels = ["A", "B", "A"]

    metrics = evaluate_prediction_sets(predictions, labels,)

    assert metrics["coverage"] == pytest.approx(2 / 3)
    assert metrics["average_set_size"] == pytest.approx(4 / 3)

    assert metrics["per_class_coverage"]["A"] == pytest.approx(0.5)
    assert metrics["per_class_coverage"]["B"] == pytest.approx(1.0)


###############################################################################################################

def test_higher_is_better_requires_unit_interval():
    df = toy_data()
    df.loc[0, "p1"] = 2.0

    model = ConformalSetModel(method="collapsed",)

    with pytest.raises(ValueError):
        model.fit(df)


##############################################################################################################

@pytest.mark.parametrize("direction, expected_message",
                         [("lower_is_better", "nonnegative"),
                          (lambda scores: -np.ones_like(scores), "nonnegative"),
                          (lambda scores: np.full_like(scores, np.inf), "finite"),
                          (lambda scores: np.full_like(scores, np.nan), "finite"),],)

########

def test_invalid_transformed_scores(direction, expected_message):
    from conformal_envelopes.missingness import transform_scores

    values = np.array([[-0.2, np.nan]])

    with pytest.raises(ValueError, match=expected_message):
        transform_scores(values, direction)

##############################################################################################################

def test_custom_transform_preserves_missing_scores():
    from conformal_envelopes.missingness import transform_scores

    values = np.array([[0.2, np.nan], [0.8, 0.4]])

    result = transform_scores(values, lambda scores: 1.0 - scores)

    np.testing.assert_allclose(result, np.array([[0.8, np.nan], [0.2, 0.6]]), equal_nan=True,)

##############################################################################################################

def test_force_nonempty_requires_a_finite_candidate():
    # Constant training scores give a predictable collapsed boundary.
    train = pd.DataFrame({"ID": [f"train_{i}" for i in range(40)], "p1": np.full(40, 0.8), 
                          "p2": np.full(40, 0.8), "label": ["A"] * 20 + ["B"] * 20,})

    model = ConformalSetModel(method="collapsed", alpha=0.1, random_state=13, force_nonempty=True,).fit(train)

    test = pd.DataFrame({"ID": ["missing", "outside"], "p1": [np.nan, 0.0], "p2": [np.nan, 0.0],})

    result = model.predict(test).set_index("ID")

    # No evidence for either candidate: do not invent a fallback.
    missing = result.loc["missing"]
    assert missing["prediction_set"] == []
    assert not bool(missing["forced"])
    assert all(np.isposinf(v) for v in missing["tau_per_label"].values())

    # Both candidates reject, but finite scores allow a fallback.
    outside = result.loc["outside"]
    assert len(outside["prediction_set"]) == 1
    assert bool(outside["forced"])

    scores = outside["tau_per_label"]
    assert all(np.isfinite(v) and v > 1.0 for v in scores.values())

    chosen = outside["prediction_set"][0]
    assert scores[chosen] == min(scores.values())


##############################################################################################################

@pytest.mark.parametrize("method,params", [("collapsed", {}),
                                            ("radial", {"n_directions": 30}), 
                                            ("strip", {"n_bins": 5, "min_samples": 3}),],)

########

def test_save_load_preserves_predictions(method, params, tmp_path):
    train = toy_data(n=100, seed=4)
    test = toy_data(n=40, seed=19)
    test["ID"] = [f"test_{i}" for i in range(len(test))]

    model = ConformalSetModel(method=method, alpha=0.1, random_state=13, force_nonempty=False,
                               **params,).fit(train)

    test_input = test.drop(columns="label")
    before = model.predict(test_input)

    path = tmp_path / f"{method}_model.pkl"
    model.save(path)
    restored = ConformalSetModel.load(path)

    after = restored.predict(test_input)

    assert before["ID"].tolist() == after["ID"].tolist()
    assert before["prediction_set"].tolist() == after["prediction_set"].tolist()
    assert before["set_size"].tolist() == after["set_size"].tolist()
    assert before["forced"].tolist() == after["forced"].tolist()

    for original_scores, restored_scores in zip(before["tau_per_label"], after["tau_per_label"]):
        assert original_scores.keys() == restored_scores.keys()
        for label in original_scores:
            np.testing.assert_allclose(original_scores[label], restored_scores[label], rtol=0, atol=0,)

##############################################################################################################

@pytest.mark.parametrize("method,params", [("collapsed", {}), 
                                           ("radial", {"n_directions": 30}),
                                           ("strip", {"n_bins": 5, "min_samples": 3}),],)

@pytest.mark.parametrize("plot_method", ["plot", "plot_slice"])

########

def test_plotting_preserves_predictions(method, params, plot_method):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train = toy_data(n=100, seed=4)
    test = toy_data(n=40, seed=19)
    test["ID"] = [f"test_{i}" for i in range(len(test))]

    model = ConformalSetModel(method=method, alpha=0.1, random_state=13,
                              force_nonempty=False, **params,).fit(train)

    test_input = test.drop(columns="label")
    before = model.predict(test_input)

    fig, ax = plt.subplots()

    try:
        returned_ax = getattr(model, plot_method)(x="p1", y="p2", label="A", test_data=test,
                                                   show_training=False, show_missing=True,
                                                   grid_size=30, ax=ax,)

        assert returned_ax is ax

        # Render the figure to catch drawing errors.
        fig.canvas.draw()

        after = model.predict(test_input)

        pd.testing.assert_frame_equal(before.drop(columns="tau_per_label"), 
                                      after.drop(columns="tau_per_label"),
                                      check_exact=True,)

        for original, following in zip(before["tau_per_label"], after["tau_per_label"]):
            assert original.keys() == following.keys()

            for label in original:
                np.testing.assert_allclose(original[label], following[label], rtol=0, atol=0,)
    finally:
        plt.close(fig)


##############################################################################################################

def test_strip_zero_limits_calibrate_consistently():
    from conformal_envelopes.envelopes.strip import (EPS, build_strip, strip_tau,
                                                      strip_is_in_region,)

    # These are already nonconformity scores.
    # The second predictor is always zero during shape discovery.
    S1 = np.tile([0.1, 0.0], (10, 1))

    # One calibration sample has a nonzero second score.
    S2 = S1.copy()
    S2[-1, 1] = 0.4

    envelope = build_strip(S1, S2, alpha=0.1, n_bins=3, min_samples=3,)

    # With 10 calibration samples, the conformal rank is 10.
    assert envelope["t_hat"] == pytest.approx(0.4 / EPS)

    # The shared scaling makes the first constraint loose.
    # The second score still determines acceptance here.
    points = np.array([[0.1, 0.0],
                       [1.0, 0.39],
                       [0.0, 0.41],])

    inside = strip_is_in_region(points, envelope)
    normalized = strip_tau(points, envelope)

    np.testing.assert_array_equal(inside, [True, True, False],)
    assert np.isfinite(normalized).all()

    np.testing.assert_array_equal(normalized <= 1.0, inside,)