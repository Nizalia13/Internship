from __future__ import annotations

from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from .io import load_table, infer_score_columns
from .missingness import missingness_mask, transform_scores
from .envelopes import (
    build_collapsed,
    collapsed_tau,
    collapsed_is_in_region,
    build_radial,
    radial_tau,
    radial_is_in_region,
    build_strip,
    strip_tau,
    strip_is_in_region,
)


_METHODS = {"collapsed", "radial", "strip"}


class ConformalSetModel:
    """General-purpose NaN-aware conformal envelope classifier.

    Expected training CSV
    ---------------------
    ID, s1, s2, ..., sk, label

    The model fits one envelope per label. At prediction time, every candidate
    label is tested against its fitted envelope and the accepted labels form
    the conformal prediction set.

    Parameters
    ----------
    method:
        "collapsed", "radial", or "strip".

    alpha:
        Target miscoverage level.

    score_direction:
        "higher_is_better" uses 1-score, matching the supplied host notebook.
        "lower_is_better" treats raw values as nonconformity scores.
        A custom callable may also be supplied.

    shape_fraction:
        Fraction of each class used for shape discovery S1. The remainder is S2
        and is used for conformal scaling.

    force_nonempty:
        If True, when no label is accepted, return the class with the smallest
        normalized tau. This reproduces the empty-set fallback used in the
        research workflow.

    random_state:
        Reproducible S1/S2 splitting and radial directions.

    Method-specific parameters
    --------------------------
    radial:
        n_directions=250
        smoothing=8.0
        angle_deg=30.0
        neighbor_fraction=0.2

    strip:
        n_bins=8
        smoothing_window=2
        min_samples=3
    """

    def __init__(
        self,
        method: str = "collapsed",
        alpha: float = 0.10,
        *,
        score_direction="higher_is_better",
        shape_fraction: float = 0.5,
        force_nonempty: bool = True,
        random_state: int = 13,
        min_class_size: int = 4,
        **method_params,
    ):
        method = method.lower()

        if method not in _METHODS:
            raise ValueError(
                f"method must be one of {sorted(_METHODS)}."
            )

        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be strictly between 0 and 1.")

        if not 0.0 < shape_fraction < 1.0:
            raise ValueError(
                "shape_fraction must be strictly between 0 and 1."
            )

        self.method = method
        self.alpha = float(alpha)
        self.score_direction = score_direction
        self.shape_fraction = float(shape_fraction)
        self.force_nonempty = bool(force_nonempty)
        self.random_state = int(random_state)
        self.min_class_size = int(min_class_size)
        self.method_params = dict(method_params)

        self.is_fitted_ = False

    def fit(
        self,
        data,
        *,
        id_col: str = "ID",
        label_col: str = "label",
        score_cols=None,
        label_to_columns: dict | None = None,
    ):
        """Fit one conformal envelope per label.

        The simple professor-style CSV needs no ``label_to_columns`` argument:
            ID | s1 | ... | sk | label

        ``label_to_columns`` is retained only as an advanced option for data
        like the original host-classification application where candidate
        labels have different score-column groups.
        """
        df = load_table(data)
        self._validate_training_frame(df, id_col, label_col)

        if score_cols is None:
            score_cols = infer_score_columns(
                df,
                id_col,
                label_col,
            )

        score_cols = list(score_cols)

        missing_cols = [
            c for c in score_cols
            if c not in df.columns
        ]
        if missing_cols:
            raise ValueError(
                f"score_cols not present in data: {missing_cols}"
            )

        numeric = df[score_cols].apply(
            pd.to_numeric,
            errors="raise",
        )

        labels = df[label_col].to_numpy()
        classes = list(pd.unique(labels))

        self.id_col_ = id_col
        self.label_col_ = label_col
        self.score_cols_ = score_cols
        self.classes_ = classes

        if label_to_columns is None:
            label_to_columns = {
                cls: list(score_cols)
                for cls in classes
            }
        else:
            label_to_columns = {
                cls: list(cols)
                for cls, cols in label_to_columns.items()
            }

            for cls in classes:
                if cls not in label_to_columns:
                    raise ValueError(
                        f"label_to_columns has no entry for {cls!r}."
                    )

                bad = [
                    c for c in label_to_columns[cls]
                    if c not in score_cols
                ]
                if bad:
                    raise ValueError(
                        f"Unknown score columns for class {cls!r}: {bad}"
                    )

        self.label_to_columns_ = label_to_columns
        self.envelopes_ = {}
        self.training_points_ = {}

        master_rng = np.random.default_rng(self.random_state)

        for cls in classes:
            class_mask = labels == cls
            n_class = int(class_mask.sum())

            if n_class < self.min_class_size:
                raise ValueError(
                    f"Class {cls!r} has only {n_class} rows. "
                    f"At least {self.min_class_size} are required."
                )

            columns = self.label_to_columns_[cls]
            raw_class = numeric.loc[
                class_mask,
                columns,
            ].to_numpy(dtype=float)

            nc_class = transform_scores(
                raw_class,
                self.score_direction,
            )

            # Match the research code's practical handling for collapsed/radial:
            # dimensions with no observations in a class carry no usable shape
            # information and are removed for that class. Strip keeps dimensions
            # so its marginal fallback logic can be applied.
            if self.method in {"collapsed", "radial"}:
                usable = ~np.isnan(nc_class).all(axis=0)
                used_columns = list(
                    np.asarray(columns)[usable]
                )
                nc_used = nc_class[:, usable]
            else:
                used_columns = list(columns)
                nc_used = nc_class

            if len(used_columns) == 0:
                raise ValueError(
                    f"Class {cls!r} has no usable score dimensions."
                )

            permutation = master_rng.permutation(n_class)
            n_shape = int(round(
                self.shape_fraction * n_class
            ))
            n_shape = min(max(n_shape, 1), n_class - 1)

            idx_s1 = permutation[:n_shape]
            idx_s2 = permutation[n_shape:]

            S1 = nc_used[idx_s1]
            S2 = nc_used[idx_s2]

            class_seed = int(
                master_rng.integers(
                    0,
                    np.iinfo(np.int32).max,
                )
            )
            class_rng = np.random.default_rng(class_seed)

            if self.method == "collapsed":
                envelope = build_collapsed(
                    S1,
                    S2,
                    self.alpha,
                )

            elif self.method == "radial":
                envelope = build_radial(
                    S1,
                    S2,
                    self.alpha,
                    n_directions=int(
                        self.method_params.get(
                            "n_directions",
                            self.method_params.get("M", 250),
                        )
                    ),
                    smoothing=float(
                        self.method_params.get(
                            "smoothing",
                            self.method_params.get("kappa", 8.0),
                        )
                    ),
                    angle_deg=float(
                        self.method_params.get(
                            "angle_deg",
                            self.method_params.get(
                                "angular_bandwidth_deg",
                                30.0,
                            ),
                        )
                    ),
                    neighbor_fraction=float(
                        self.method_params.get(
                            "neighbor_fraction",
                            self.method_params.get(
                                "neighbor_frac",
                                0.2,
                            ),
                        )
                    ),
                    rng=class_rng,
                )

            else:
                envelope = build_strip(
                    S1,
                    S2,
                    self.alpha,
                    n_bins=int(
                        self.method_params.get(
                            "n_bins",
                            self.method_params.get(
                                "number_of_bins",
                                self.method_params.get("NB", 8),
                            ),
                        )
                    ),
                    smoothing_window=int(
                        self.method_params.get(
                            "smoothing_window",
                            self.method_params.get(
                                "monotonic_window",
                                2,
                            ),
                        )
                    ),
                    min_samples=int(
                        self.method_params.get(
                            "min_samples",
                            3,
                        )
                    ),
                )

            self.envelopes_[cls] = {
                "label": cls,
                "method": self.method,
                "columns": used_columns,
                "alpha": self.alpha,
                "shape_size": len(S1),
                "calibration_size": len(S2),
                "parameters": dict(self.method_params),
                "envelope": envelope,
            }

            self.training_points_[cls] = pd.DataFrame(
                raw_class,
                columns=columns,
            )

        self.fitted_classes_ = list(self.envelopes_)
        self.is_fitted_ = True
        return self

    def _validate_training_frame(
        self,
        df,
        id_col,
        label_col,
    ):
        if id_col not in df.columns:
            raise ValueError(
                f"ID column {id_col!r} was not found."
            )

        if label_col not in df.columns:
            raise ValueError(
                f"Label column {label_col!r} was not found."
            )

        if df[id_col].isna().any():
            raise ValueError("ID column contains missing values.")

        if df[id_col].duplicated().any():
            raise ValueError("Training IDs must be unique.")

        if df[label_col].isna().any():
            raise ValueError(
                "Training labels contain missing values."
            )

    def _check_fitted(self):
        if not self.is_fitted_:
            raise RuntimeError("Call fit() before this method.")

    def _class_tau_and_membership(
        self,
        score_frame,
        cls,
    ):
        info = self.envelopes_[cls]
        columns = info["columns"]
        envelope = info["envelope"]

        raw = score_frame[columns].to_numpy(dtype=float)
        nc = transform_scores(
            raw,
            self.score_direction,
        )

        if self.method == "collapsed":
            tau = collapsed_tau(nc, envelope)
            inside = collapsed_is_in_region(
                nc,
                envelope,
            )

        elif self.method == "radial":
            tau = radial_tau(nc, envelope)
            inside = radial_is_in_region(
                nc,
                envelope,
            )

        else:
            tau = strip_tau(nc, envelope)
            inside = strip_is_in_region(
                nc,
                envelope,
            )

        return tau, inside

    def predict(
        self,
        data,
        *,
        output: str = "dataframe",
        candidate_labels=None,
    ):
        """Return a conformal set for every test ID.

        Prediction data must contain the ID and predictor score columns.
        It may also contain a label column; labels are ignored during prediction.
        """
        self._check_fitted()
        df = load_table(data)

        if self.id_col_ not in df.columns:
            raise ValueError(
                f"ID column {self.id_col_!r} was not found."
            )

        if df[self.id_col_].isna().any():
            raise ValueError(
                "Prediction IDs contain missing values."
            )

        if df[self.id_col_].duplicated().any():
            raise ValueError(
                "Prediction IDs must be unique."
            )

        needed = sorted({
            column
            for info in self.envelopes_.values()
            for column in info["columns"]
        })

        missing = [
            c for c in needed
            if c not in df.columns
        ]
        if missing:
            raise ValueError(
                f"Prediction data is missing score columns: {missing}"
            )

        score_frame = df[needed].apply(
            pd.to_numeric,
            errors="raise",
        )

        if candidate_labels is None:
            active_classes = list(self.fitted_classes_)
        else:
            active_classes = [
                cls for cls in candidate_labels
                if cls in self.envelopes_
            ]

            if not active_classes:
                raise ValueError(
                    "No requested candidate labels have fitted envelopes."
                )

        tau_by_label = {}
        inside_by_label = {}

        for cls in active_classes:
            tau, inside = self._class_tau_and_membership(
                score_frame,
                cls,
            )
            tau_by_label[cls] = tau
            inside_by_label[cls] = inside

        records = []

        for i in range(len(df)):
            prediction_set = [
                cls
                for cls in active_classes
                if bool(inside_by_label[cls][i])
            ]

            forced = False

            if len(prediction_set) == 0 and self.force_nonempty:
                forced = True
                best = min(
                    active_classes,
                    key=lambda cls: float(
                        tau_by_label[cls][i]
                    ),
                )
                prediction_set = [best]

            records.append({
                self.id_col_: df.iloc[i][self.id_col_],
                "prediction_set": prediction_set,
                "set_size": len(prediction_set),
                "forced": forced,
                "tau_per_label": {
                    cls: float(tau_by_label[cls][i])
                    for cls in active_classes
                },
            })

        if output == "dataframe":
            return pd.DataFrame(records)

        if output == "dict":
            return {
                row[self.id_col_]: row["prediction_set"]
                for row in records
            }

        if output == "records":
            return records

        raise ValueError(
            "output must be 'dataframe', 'dict', or 'records'."
        )

    def get_envelopes(self):
        """Return the fitted envelope structure, keyed by label."""
        self._check_fitted()
        return self.envelopes_

    def missingness_mask(self, data):
        """Return 1=observed, 0=missing for all original score columns."""
        self._check_fitted()
        df = load_table(data)

        missing = [
            c for c in self.score_cols_
            if c not in df.columns
        ]
        if missing:
            raise ValueError(
                f"Data is missing score columns: {missing}"
            )

        values = df[self.score_cols_].to_numpy(dtype=float)
        mask = missingness_mask(values)

        return pd.DataFrame(
            mask,
            columns=self.score_cols_,
            index=df.index,
        )

    def save(self, path):
        """Save the fitted model, including all learned envelopes."""
        self._check_fitted()

        if callable(self.score_direction):
            raise ValueError(
                "Models with a custom callable transformation are not "
                "serialized by save(). Use a named score_direction or "
                "manage callable serialization yourself."
            )

        path = Path(path)

        with path.open("wb") as file:
            pickle.dump(self, file)

        return path

    @classmethod
    def load(cls, path):
        """Load a model created with save()."""
        with Path(path).open("rb") as file:
            model = pickle.load(file)

        if not isinstance(model, cls):
            raise TypeError(
                "The saved object is not a ConformalSetModel."
            )

        return model

    def plot(
        self,
        *,
        x,
        y,
        label,
        grid_size: int = 160,
        show_training: bool = True,
        ax=None,
    ):
        """Plot a two-score view of one fitted label envelope."""
        self._check_fitted()

        if label not in self.envelopes_:
            raise ValueError(
                f"No fitted envelope exists for label {label!r}."
            )

        return _plot_2d(
            self,
            x=x,
            y=y,
            label=label,
            grid_size=grid_size,
            show_training=show_training,
            ax=ax,
        )

    # Friendly alias if users prefer an explicit name.
    plot_2d = plot


def _plot_2d(
    model,
    *,
    x,
    y,
    label,
    grid_size,
    show_training,
    ax,
):
    import matplotlib.pyplot as plt

    info = model.envelopes_[label]
    columns = info["columns"]

    if x == y:
        raise ValueError("x and y must be different columns.")

    if x not in columns or y not in columns:
        raise ValueError(
            f"x and y must be score columns used by label {label!r}. "
            f"Available columns: {columns}"
        )

    training = model.training_points_[label]

    xv = training[x].to_numpy(dtype=float)
    yv = training[y].to_numpy(dtype=float)

    finite_x = xv[np.isfinite(xv)]
    finite_y = yv[np.isfinite(yv)]

    if len(finite_x) == 0 or len(finite_y) == 0:
        raise ValueError(
            "Selected plotting columns contain no finite training values."
        )

    def axis_bounds(values):
        lo = float(np.min(values))
        hi = float(np.max(values))
        span = hi - lo
        pad = 0.08 * span if span > 0 else 0.1
        return lo - pad, hi + pad

    xlo, xhi = axis_bounds(finite_x)
    ylo, yhi = axis_bounds(finite_y)

    gx = np.linspace(xlo, xhi, int(grid_size))
    gy = np.linspace(ylo, yhi, int(grid_size))
    XX, YY = np.meshgrid(gx, gy)

    probe = pd.DataFrame(
        np.nan,
        index=np.arange(XX.size),
        columns=columns,
    )
    probe[x] = XX.ravel()
    probe[y] = YY.ravel()

    tau, inside = model._class_tau_and_membership(
        probe,
        label,
    )

    ZZ = inside.astype(float).reshape(XX.shape)

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    ax.contour(
        XX,
        YY,
        ZZ,
        levels=[0.5],
        linewidths=2,
    )

    if show_training:
        ax.scatter(
            xv,
            yv,
            s=20,
            alpha=0.7,
        )

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(
        f"{model.method.capitalize()} envelope for {label}"
    )

    return ax
