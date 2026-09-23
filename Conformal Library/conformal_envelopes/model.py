# io --> missingness --> collapsed/radial/strip --> model (coordinator)
# The main interface: fits one envelope per class, predicts label sets, saves/loads models, and plots results.

from __future__ import annotations

from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from .io import load_table, infer_score_columns
from .missingness import missingness_mask, transform_scores

# brings in all three envelope methods.
from .envelopes import (build_collapsed, collapsed_tau, collapsed_is_in_region, build_radial,
                        radial_tau, radial_is_in_region, build_strip, strip_tau, strip_is_in_region,)

_METHODS = {"collapsed", "radial", "strip"}

# this class contains configuration, fitting, prediction, saving, loading, plotting, helper
class ConformalSetModel:
    """
       General-purpose NaN-aware conformal envelope classifier.

       Expected training CSV
             ---------------------
           ID, s1, s2, ..., sk, label

       It fits one envelope per label. At prediction time, every candidate label is tested against its 
       fitted envelope and the accepted labels form the conformal prediction set.

       ----------
       Parameters
       ----------
       - method:
            "collapsed", "radial", or "strip".

       - alpha:
            Target miscoverage level.

        - score_direction:
            - "higher_is_better": uses 1-score, matching the supplied host notebook.
            - "lower_is_better": treats raw values as nonconformity scores.
            - A custom callable may also be supplied.

        - shape_fraction:
            Fraction of each class used for shape discovery S1. The remainder is S2 and is used for conformal scaling.

        - force_nonempty:
            If True and no label is accepted, select the candidate with the smallest finite normalized tau.
            If no finite candidate exists, leave the set empty.

        - random_state:
            Reproducible S1/S2 splitting and radial directions.

        ---------------------------
        Method-specific parameters
        ---------------------------
        - radial:
            - n_directions=250
            - smoothing=8.0
            - angle_deg=30.0
            - neighbor_fraction=0.2

        - strip:
            - n_bins=8
            - min_samples=3
    """

    # This function runs when we do: "model = "ConformalSetModel(...)"
    # doesnt fit anything, onlu stores the settings
    def __init__(self, method: str = "collapsed", alpha: float = 0.10, *, score_direction="higher_is_better",
                 shape_fraction: float = 0.5, force_nonempty: bool = True, random_state: int = 13, 
                 min_class_size: int = 3, **method_params,):
        method = method.lower()

        if method not in _METHODS:
            raise ValueError(f"method must be one of {sorted(_METHODS)}.")

        # validating alpha value
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be strictly between 0 and 1.")

        # the percentage of data that goes to S1 from each class
        if not 0.0 < shape_fraction < 1.0:
            raise ValueError("shape_fraction must be strictly between 0 and 1.")

        # storing settings - all of these become attributes of the model
        self.method = method
        self.alpha = float(alpha)
        self.score_direction = score_direction
        self.shape_fraction = float(shape_fraction)
        self.force_nonempty = bool(force_nonempty)
        self.random_state = int(random_state)
        self.min_class_size = int(min_class_size)
        allowed_params = {"collapsed": set(),
                          "radial": {"n_directions", "M", "smoothing", "kappa", "angle_deg", "angular_bandwidth_deg",
                                     "neighbor_fraction", "neighbor_frac",},
                          "strip": {"n_bins", "number_of_bins", "NB", "min_samples",},}

        unknown = set(method_params) - allowed_params[method]
        if unknown:
            raise ValueError(f"Unknown parameters for {method!r}: {sorted(unknown)}. "
                             f"Allowed parameters: {sorted(allowed_params[method])}")

        # Reject competing names for the same parameter.
        alias_groups = {"collapsed": [],
                        "radial": [("n_directions", "M"), ("smoothing", "kappa"), ("angle_deg", "angular_bandwidth_deg"),
                                   ("neighbor_fraction", "neighbor_frac"),],
                        "strip": [("n_bins", "number_of_bins", "NB"),],}

        for aliases in alias_groups[method]:
            supplied = [name for name in aliases if name in method_params]
            if len(supplied) > 1:
                raise ValueError(f"Supply only one of {aliases}; received {supplied}.")

        self.method_params = dict(method_params)

        # this model exists, but fit() has not been called yet.
        self.is_fitted_ = False

#################################################################################################################################3
    
    # the model actually learns the envelopes.
    def fit(self, data, *, id_col: str = "ID", label_col: str = "label", score_cols=None,
            label_to_columns: dict | None = None,):
        """
           Fit one conformal envelope per label.
                   ID | s1 | ... | sk | label

           ``label_to_columns`` is retained only as an option where candidate labels have different score-column groups.
        """

        # If fitting fails halfway through, the model should not pretend it is fitted.
        self.is_fitted_ = False
        df = load_table(data)
        self._validate_training_frame(df, id_col, label_col)

        if score_cols is None:
            score_cols = infer_score_columns(df, id_col, label_col,)

        score_cols = list(score_cols)

        missing_cols = [c for c in score_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"score_cols not present in data: {missing_cols}")

        numeric = df[score_cols].apply(pd.to_numeric, errors="raise",)

        labels = df[label_col].to_numpy()
        classes = sorted(pd.unique(labels), key=str)

        self.id_col_ = id_col
        self.label_col_ = label_col
        self.score_cols_ = score_cols
        self.classes_ = classes

        # If a custom mapping is supplied, it converts every entry to a list.
        # Then it checks:
        #     - every class has an entry
        #     - every listed column actually exists
        if label_to_columns is None:
            label_to_columns = {cls: list(score_cols) for cls in classes}
        else:
            label_to_columns = {cls: list(cols) for cls, cols in label_to_columns.items()}

            for cls in classes:
                if cls not in label_to_columns:
                    raise ValueError(f"label_to_columns has no entry for {cls!r}.")

                bad = [c for c in label_to_columns[cls] if c not in score_cols]
                if bad:
                    raise ValueError(f"Unknown score columns for class {cls!r}: {bad}")

        self.label_to_columns_ = label_to_columns
        self.envelopes_ = {}
        self.training_points_ = {}

        master_rng = np.random.default_rng(self.random_state)

        # one-envelope-per-label
        for cls in classes:
            class_mask = labels == cls
            n_class = int(class_mask.sum())

            # each label needs enough data
            if n_class < self.min_class_size:
                raise ValueError(f"Class {cls!r} has only {n_class} rows. "
                                 f"At least {self.min_class_size} are required.")

            # extracts only:
            #     - rows belonging to this class
            #     - columns assigned to this class
            columns = self.label_to_columns_[cls]
            raw_class = numeric.loc[class_mask, columns,].to_numpy(dtype=float)

            # gets the nonconformity scores (lower values become better nonconformity scores.)
            # raw scores --> transform scores --> NC scores --> all envelopes
            nc_class = transform_scores(raw_class, self.score_direction,)
 
            # checks which columns are not entirely NaN for this class.
            if self.method in {"collapsed", "radial"}:
                usable = ~np.isnan(nc_class).all(axis=0)
                used_columns = list(np.asarray(columns)[usable])
                nc_used = nc_class[:, usable]
            else:
                used_columns = list(columns)
                nc_used = nc_class

            # Ensure at least one usable dimension
            if len(used_columns) == 0:
                raise ValueError(f"Class {cls!r} has no usable score dimensions.")

            permutation = master_rng.permutation(n_class)
            n_shape = int(self.shape_fraction * n_class)
            n_shape = min(max(n_shape, 1), n_class - 1)

            idx_s1 = permutation[:n_shape]
            idx_s2 = permutation[n_shape:]

            # shape/calibration split.
            S1 = nc_used[idx_s1]
            S2 = nc_used[idx_s2]

            # class_seed = int(master_rng.integers(0, np.iinfo(np.int32).max,))
            # class_rng = np.random.default_rng(class_seed)

            # Which envelope geometry are we using?
            if self.method == "collapsed":
                envelope = build_collapsed(S1, S2, self.alpha,)

            elif self.method == "radial":
                envelope = build_radial(S1, S2, self.alpha, 
                                        n_directions=self.method_params.get("n_directions", 
                                        self.method_params.get("M", 250)), 
                                        smoothing=self.method_params.get("smoothing", 
                                                                         self.method_params.get("kappa", 8.0)),
                                        angle_deg=self.method_params.get("angle_deg",
                                                                         self.method_params.get("angular_bandwidth_deg", 30.0),),
                                        neighbor_fraction=self.method_params.get("neighbor_fraction", 
                                                                                 self.method_params.get("neighbor_frac", 0.2),),
                                        rng=master_rng,)

            else:
                envelope = build_strip(S1, S2, self.alpha, 
                                       n_bins=self.method_params.get("n_bins", 
                                                                     self.method_params.get("number_of_bins", 
                                                                                            self.method_params.get("NB", 8)),),
                                       min_samples=self.method_params.get("min_samples", 3),)

            # for each class we store which label it belongs to, which geometry was used, which score dmensions where used, the
            # miscoverage level, how many points used, ow many used for conformal envelope, method
            self.envelopes_[cls] = {"label": cls, "method": self.method, "columns": used_columns, "alpha": self.alpha, 
                                    "shape_size": len(S1), "calibration_size": len(S2), "parameters": dict(self.method_params),
                                    "idx_s1": idx_s1.copy(),"idx_s2": idx_s2.copy(),"envelope": envelope,}

            # raw scores, not transformed nonconformity scores.
            self.training_points_[cls] = pd.DataFrame(raw_class, columns=columns,)

        # stores which classes successfully got fitted.
        self.fitted_classes_ = list(self.envelopes_)
        self.is_fitted_ = True
        return self

#################################################################################################################################3

    # checks basic dataset structure
    def _validate_training_frame(self, df, id_col, label_col,):
        if df.empty:
            # reject empty training data.
            raise ValueError("Training data must contain at least one sample.")

        if id_col not in df.columns:
            # require ID column.
            raise ValueError(f"ID column {id_col!r} was not found.")

        if label_col not in df.columns:
            # require label column.
            raise ValueError(f"Label column {label_col!r} was not found.")

        if df[id_col].isna().any():
            # IDs cannot be missing.
            raise ValueError("ID column contains missing values.")

        if df[id_col].duplicated().any():
            # IDs must be unique.
            raise ValueError("Training IDs must be unique.")

        if df[label_col].isna().any():
            # training labels cannot be missing.
            raise ValueError("Training labels contain missing values.")

#################################################################################################################################3

    # So someone cannot do: "model.predict(test)" before: "model.fit(train)"
    def _check_fitted(self):
        if not self.is_fitted_:
            raise RuntimeError("Call fit() before this method.")

#################################################################################################################################3

    # for one candidate class, calculate tau and whether every test sample is inside that class envelope.
    def _class_tau_and_membership(self, score_frame, cls):
        info = self.envelopes_[cls]
        envelope = info["envelope"]

        raw = score_frame[info["columns"]].to_numpy(dtype=float)
        nc = transform_scores(raw, self.score_direction)

        # Strip evaluates only the columns retained during fitting.
        effective_scores = (nc[:, envelope["keep_columns"]] if self.method == "strip" else nc)

        missing = np.isnan(effective_scores).all(axis=1)
        usable = ~missing

        # Match the notebook's method-specific missing-score behavior.
        if self.method == "collapsed":
            tau = np.full(len(nc), np.inf)
            inside = np.zeros(len(nc), dtype=bool)
        else:
            tau = np.zeros(len(nc), dtype=float)
            inside = np.ones(len(nc), dtype=bool)

        if not usable.any():
            return tau, inside

        scores = nc[usable]

        if self.method == "collapsed":
            tau[usable] = collapsed_tau(scores, envelope)
            inside[usable] = collapsed_is_in_region(scores, envelope)

        elif self.method == "radial":
            tau[usable] = radial_tau(scores, envelope)
            inside[usable] = tau[usable] <= 1.0

        else:
            tau[usable] = strip_tau(scores, envelope)
            inside[usable] = strip_is_in_region(scores, envelope)

        return tau, inside

#################################################################################################################################3

    def predict(self, data, *, output: str = "dataframe", candidate_labels=None,):
        """
           Returns a conformal set for every test ID.

           Prediction data must contain the ID and predictor score columns.
           It may also contain a label column; labels are ignored during prediction.
        """
        self._check_fitted()
        df = load_table(data)

        if self.id_col_ not in df.columns:
            # ID column exists
            raise ValueError(f"ID column {self.id_col_!r} was not found.")

        if df[self.id_col_].isna().any():
            # IDs are not missing
            raise ValueError("Prediction IDs contain missing values.")

        if df[self.id_col_].duplicated().any():
            # IDs are unique
            raise ValueError("Prediction IDs must be unique.")

        if candidate_labels is None:
            active_classes = list(self.fitted_classes_)
        else:
            if isinstance(candidate_labels, (str, bytes)):
                raise ValueError("candidate_labels must be a list of labels, not a single string.")

            active_classes = list(dict.fromkeys(candidate_labels))

            unknown = [cls for cls in active_classes if cls not in self.envelopes_]
            if unknown:
                raise ValueError(f"Unknown candidate labels: {unknown}")

            if not active_classes:
                raise ValueError("Provide at least one candidate label.")

        # Require only columns used by the requested classes.
        needed = list(dict.fromkeys(column for cls in active_classes 
                                    for column in self.envelopes_[cls]["columns"]))

        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise ValueError(f"Prediction data is missing score columns: {missing}")

        score_frame = df[needed].apply(pd.to_numeric, errors="raise")

        tau_by_label = {}
        inside_by_label = {}

        for cls in active_classes:
            tau, inside = self._class_tau_and_membership(score_frame, cls,)
            tau_by_label[cls] = tau
            inside_by_label[cls] = inside

        # Build prediction set for each sample
        records = []

        for i in range(len(df)):
            prediction_set = [cls for cls in active_classes if bool(inside_by_label[cls][i])]

            forced = False

            # if len(prediction_set) == 0 and self.force_nonempty:
            #     forced = True
            #     best = min(active_classes, key=lambda cls: float(tau_by_label[cls][i]),)
            #     prediction_set = [best]

            if len(prediction_set) == 0 and self.force_nonempty:
                finite_candidates = [cls for cls in active_classes if np.isfinite(tau_by_label[cls][i])]

                if finite_candidates:
                    best = min(finite_candidates, key=lambda cls: float(tau_by_label[cls][i]),)
                    prediction_set = [best]
                    forced = True



            records.append({self.id_col_: df.iloc[i][self.id_col_], "prediction_set": prediction_set,
                            "set_size": len(prediction_set), "forced": forced,
                            "tau_per_label": {cls: float(tau_by_label[cls][i]) for cls in active_classes},})

        if output == "dataframe":
            return pd.DataFrame(records)

        if output == "dict":
            return {row[self.id_col_]: row["prediction_set"] for row in records}

        if output == "records":
            return records

        raise ValueError("output must be 'dataframe', 'dict', or 'records'.")

#################################################################################################################################3

    def predict_per_sample(self, data, *, candidates_by_id):
        """
           Predict with candidate labels supplied separately for each ID.

          Samples sharing the same ordered candidate list are evaluated
          together. Output rows follow the original input order.
        """
        self._check_fitted()
        df = load_table(data)
        id_col = self.id_col_

        output_columns = [id_col, "prediction_set", "set_size","forced","tau_per_label",]

        if id_col not in df.columns:
            raise ValueError(f"ID column {id_col!r} was not found.")

        ids = df[id_col]

        if ids.isna().any() or ids.duplicated().any():
            raise ValueError("Prediction IDs must be nonmissing and unique.")

        if df.empty:
            return pd.DataFrame(columns=output_columns)

        groups = {}

        # Validate candidate lists before evaluating any group.
        for position, sample_id in enumerate(ids):
            if sample_id not in candidates_by_id:
                raise ValueError(f"No candidate labels supplied for ID {sample_id!r}.")

            supplied = candidates_by_id[sample_id]

            if isinstance(supplied, (str, bytes)):
                raise ValueError(f"Candidates for ID {sample_id!r} must be a "
                                 "sequence of labels, not a single string.")

            candidates = tuple(dict.fromkeys(supplied))

            unknown = [label for label in candidates if label not in self.envelopes_]
            if unknown:
                raise ValueError(f"Unknown candidate labels for ID " f"{sample_id!r}: {unknown}")

            groups.setdefault(candidates, []).append(position)

        outputs = []

        for candidates, positions in groups.items():
            batch = df.iloc[positions]

            if candidates:
                result = self.predict(batch, candidate_labels=list(candidates), output="dataframe",)
            else:
                # No permitted candidates: do not force a label.
                result = pd.DataFrame({id_col: batch[id_col].tolist(), "prediction_set": [[] for _ in positions],
                                       "set_size": [0] * len(positions), "forced": [False] * len(positions), 
                                       "tau_per_label": [{} for _ in positions],})

            outputs.append(result)

        combined = pd.concat(outputs, ignore_index=True)

        return (combined.set_index(id_col).loc[ids.tolist()].reset_index()[output_columns])

#################################################################################################################################3

    def get_envelopes(self):
        """
           Return the fitted envelope structure, given by label.
        """
        self._check_fitted()
        return self.envelopes_

#################################################################################################################################3
   
    # inspect missingness in the original score space.
    # It loads the data, checks all original score columns exist, gets their values, then calls missingness_mask
    def missingness_mask(self, data):
        """
           Return:
              - 1 = observed
              - 0 = missing 
              for all original score columns.
        """
        self._check_fitted()
        df = load_table(data)

        missing = [c for c in self.score_cols_ if c not in df.columns]
        if missing:
            raise ValueError(f"Data is missing score columns: {missing}")

        values = df[self.score_cols_].to_numpy(dtype=float)
        mask = missingness_mask(values)

        return pd.DataFrame(mask, columns=self.score_cols_, index=df.index,)

#################################################################################################################################3

    def save(self, path):
        """
           Save the fitted model, including all learned envelopes.
        """
        self._check_fitted()

        if callable(self.score_direction):
            raise ValueError("Models with a custom callable transformation are not serialized by save()."
                             "Use a named score_direction or manage callable serialization yourself.")

        path = Path(path)

        with path.open("wb") as file:
            pickle.dump(self, file)

        return path

#################################################################################################################################3

    @classmethod
    def load(cls, path):
        """
           Load a model created with save().
        """
        with Path(path).open("rb") as file:
            model = pickle.load(file)

        if not isinstance(model, cls):
            raise TypeError("The saved object is not a ConformalSetModel.")
        return model

#################################################################################################################################3

    def plot(self, *, x, y, label, test_data=None, grid_size=160, show_training=False,
             show_other_classes=False, show_missing=True, ax=None,):
        """
           Plot a separate 2D diagnostic envelope and optional test points.

           Test membership refers to this 2D envelope, not the full model.
           Test data is never used to fit or calibrate the envelope.
        """
        from .plotting import _plot_2d
        self._check_fitted()

        if label not in self.envelopes_:
            raise ValueError(f"No fitted envelope exists for label {label!r}.")

        return _plot_2d(self, x=x, y=y, label=label, test_data=test_data, grid_size=grid_size,
                        show_training=show_training, show_other_classes=show_other_classes,
                        show_missing=show_missing, ax=ax,)

    plot_2d = plot


#################################################################################################################################3

    def plot_slice(self, *, x, y, label, test_data, show_training=False, show_missing=True, 
                   grid_size=160, ax=None,):
        """
           Plot a slice of the existing fitted envelope without refitting.

           Unplotted coordinates are fixed at training median nonconformity values. Test colours show full-model membership.
        """
        from .plotting import _plot_envelope_slice
        self._check_fitted()

        return _plot_envelope_slice(self, x=x, y=y, label=label, test_data=test_data, show_training=show_training,
                                    show_missing=show_missing, grid_size=grid_size, ax=ax,)

#################################################################################################################################3

