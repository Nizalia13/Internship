from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from .io import load_table
from .missingness import transform_scores
from .envelopes import (build_collapsed, collapsed_is_in_region, build_radial, radial_is_in_region,
                        build_strip, strip_is_in_region,)

#######################################################################################################

def _plot_2d(model, *, x, y, label, test_data, grid_size, show_training, show_other_classes, 
             show_missing, ax,):
    if (isinstance(grid_size, (bool, np.bool_)) or not isinstance(grid_size, (int, np.integer)) 
    or grid_size < 2):
        raise ValueError("grid_size must be an integer >= 2.")

    info = model.envelopes_[label]
    columns = info["columns"]

    if x == y:
        raise ValueError("x and y must be different columns.")

    if x not in columns or y not in columns:
        raise ValueError(f"x and y must be columns used by label {label!r}. "
                         f"Available columns: {columns}")

    def to_nc(frame):
        missing = [column for column in (x, y) if column not in frame.columns]
        if missing:
            raise ValueError(f"Plotting data is missing columns: {missing}")

        raw = frame[[x, y]].to_numpy(dtype=float)
        scores = transform_scores(raw, model.score_direction)

        observed = scores[~np.isnan(scores)]
        if not np.isfinite(observed).all() or (observed < 0).any():
            raise ValueError("Plotting requires finite, nonnegative "
                             "nonconformity scores, or NaN.")
        return scores

    # Only saved training data is used to construct the envelope.
    train_nc = to_nc(model.training_points_[label])

    S1 = train_nc[info["idx_s1"]]
    S2 = train_nc[info["idx_s2"]]

    # The 2D diagnostic excludes training rows missing both coordinates.
    S1 = S1[~np.isnan(S1).all(axis=1)]
    S2 = S2[~np.isnan(S2).all(axis=1)]

    if len(S1) == 0 or len(S2) == 0:
        raise ValueError("Cannot plot this pair: S1 and S2 each need samples "
                         "with at least one selected score observed.")

    if np.isnan(S1).all(axis=0).any():
        raise ValueError("Cannot fit a 2D envelope: a selected column has no observed values in S1.")

    params = model.method_params

    if model.method == "collapsed":
        envelope = build_collapsed(S1, S2, model.alpha)
        membership = collapsed_is_in_region

    elif model.method == "radial":
        envelope = build_radial(S1, S2, model.alpha, n_directions=params.get("n_directions", params.get("M", 250)),
                                smoothing=params.get("smoothing", params.get("kappa", 8.0)), 
                                angle_deg=params.get("angle_deg", params.get("angular_bandwidth_deg", 30.0)),
                                neighbor_fraction=params.get("neighbor_fraction", params.get("neighbor_frac", 0.2)),
                                rng=np.random.default_rng(model.random_state),)
        membership = radial_is_in_region

    else:
        envelope = build_strip(S1, S2, model.alpha, n_bins=params.get("n_bins", params.get("number_of_bins", 
                                                                                           params.get("NB", 8))), 
                               min_samples=params.get("min_samples", 3),)
        membership = strip_is_in_region

    test_frame = (load_table(test_data) if test_data is not None else None)

    test_nc = (to_nc(test_frame) if test_frame is not None else np.empty((0, 2), dtype=float))

    test_truth = None

    if test_frame is not None and model.label_col_ in test_frame.columns:
        if test_frame[model.label_col_].isna().any():
            raise ValueError("Test labels contain missing values. Supply complete " 
                             "labels, or omit the label column to plot acceptance only.")

        test_truth = test_frame[model.label_col_].to_numpy()


    # This existing option controls OTHER CLASSES' TRAINING POINTS.
    other_training = []
    if show_training and show_other_classes:
        for other_label, frame in model.training_points_.items():
            if other_label == label:
                continue

            if x not in frame.columns or y not in frame.columns:
                raise ValueError(f"Stored training data for {other_label!r} does not "
                                 f"contain {x!r} and {y!r}. Use show_other_classes=False. "
                                 "Test points from all classes can still be supplied through test_data.")

            other_training.append((other_label, to_nc(frame)))

    # Test coordinates affect the display range, never the envelope fit.
    bounds_points = [train_nc, test_nc]
    bounds_points.extend(points for _, points in other_training)
    bounds_points = np.concatenate(bounds_points, axis=0)

    def upper_bound(values):
        finite = values[np.isfinite(values)]
        hi = float(finite.max())
        return hi + (0.08 * hi if hi > 0 else 0.1)

    xhi = upper_bound(bounds_points[:, 0])
    yhi = upper_bound(bounds_points[:, 1])

    gx = np.linspace(0.0, xhi, grid_size)
    gy = np.linspace(0.0, yhi, grid_size)
    XX, YY = np.meshgrid(gx, gy)

    # Already in nonconformity space: no second transformation.
    grid_nc = np.column_stack((XX.ravel(), YY.ravel()))
    grid_inside = membership(grid_nc, envelope)
    ZZ = grid_inside.reshape(XX.shape).astype(float)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    ax.contourf(XX, YY, ZZ, levels=[0.5, 1.5], colors=["#b8dfba"], alpha=0.4,)

    if grid_inside.any() and not grid_inside.all():
        ax.contour(XX, YY, ZZ, levels=[0.5], colors=["#482878"], linewidths=2,)

    def draw_points(points, name, color, marker, alpha):
        complete = np.isfinite(points).all(axis=1)

        if complete.any():
            ax.scatter(points[complete, 0], points[complete, 1], s=32, color=color, marker=marker,
                       alpha=alpha, label=name, zorder=3,)

        if show_missing:
            x_only = np.isfinite(points[:, 0]) & np.isnan(points[:, 1])
            y_only = np.isnan(points[:, 0]) & np.isfinite(points[:, 1])

            for index, value in enumerate(points[x_only, 0]):
                ax.axvline(value, color=color, linestyle=":", linewidth=1, alpha=0.35,
                           label=("_nolegend_"),)

            for index, value in enumerate(points[y_only, 1]):
                ax.axhline(value, color=color, linestyle=":", linewidth=1, alpha=0.35,
                           label=("_nolegend_"),)

    if show_training:
        draw_points(train_nc, f"Training: {label}", "gray", ".", 0.35)
        for other_label, points in other_training:
            draw_points(points, f"Training: {other_label}", "silver", "x", 0.4)

    notes = []

    if test_data is not None:
        usable = ~np.isnan(test_nc).all(axis=1)
        test_inside = np.zeros(len(test_nc), dtype=bool)

        # Do not pass completely missing rows to envelope functions.
        if usable.any():
            test_inside[usable] = membership(test_nc[usable], envelope)

        accepted = usable & test_inside
        rejected = usable & ~test_inside

        if test_truth is None:
            # Unlabelled test data: show acceptance only.
            draw_points(test_nc[accepted], "Test: accepted", "tab:green", "o", 0.8,)
            draw_points(test_nc[rejected], "Test: rejected", "tab:red", "x", 0.9,)

        else:
            belongs_to_label = test_truth == label

            true_accept = accepted & belongs_to_label
            false_reject = rejected & belongs_to_label
            false_accept = accepted & ~belongs_to_label
            true_reject = rejected & ~belongs_to_label

            draw_points(test_nc[true_accept], "Test: true accept", "tab:green", "o", 0.85,)
            draw_points(test_nc[false_reject], "Test: false reject", "tab:red", "x", 0.95,)
            draw_points(test_nc[false_accept], "Test: false accept", "tab:orange", "^", 0.85,)
            draw_points(test_nc[true_reject], "Test: true reject", "tab:blue", "x", 0.65,)

        both_missing = int((~usable).sum())
        # if both_missing:
        #     notes.append(f"{both_missing} test rows missing both coordinates "
        #                  "(not drawn or evaluated)")

        if not show_missing:
            partial = usable & np.isnan(test_nc).any(axis=1)
            if partial.any():
                notes.append(f"{partial.sum()} partially observed test rows "
                             "evaluated but not drawn")

    if grid_inside.all():
        notes.append("All grid points accepted; no boundary in view")
    elif not grid_inside.any():
        notes.append("All grid points rejected; no boundary in view")

    if notes:
        ax.text(0.02, 0.98, "\n".join(notes), transform=ax.transAxes, ha="left", va="top", 
                fontsize=8, bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"), zorder=5,)

    region_handle = Patch(facecolor="#b8dfba", alpha=0.4, label=f"2D accepted region for {label}",)
    handles, labels = ax.get_legend_handles_labels()

    # Add one shared explanation for all dotted missing-score lines.
    has_missing_lines = any(line.get_linestyle() == ":" for line in ax.lines)

    if has_missing_lines:
        handles.append(Line2D([0], [0], color="gray", linestyle=":", linewidth=1.5,))
        labels.append("One score missing")

    ax.legend([region_handle] + handles, [region_handle.get_label()] + labels, title="Test categories: 2D decisions",
               fontsize=8, title_fontsize=8, loc="best",)

    ax.set_xlim(0.0, xhi)
    ax.set_ylim(0.0, yhi)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"{model.method.capitalize()}: 2D envelope for {label}")

    return ax

#############################################################################################################################################

def _plot_envelope_slice(model, *, x, y, label, test_data, show_training=False, show_missing=True, 
                         grid_size=160, ax=None,):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    if label not in model.envelopes_:
        raise ValueError(f"No fitted envelope for {label!r}.")

    if (isinstance(grid_size, (bool, np.bool_)) or not isinstance(grid_size, (int, np.integer))
        or grid_size < 2):
        raise ValueError("grid_size must be an integer >= 2.")

    info = model.envelopes_[label]
    envelope = info["envelope"]
    columns = list(info["columns"])

    if x == y or x not in columns or y not in columns:
        raise ValueError("Choose two different score columns used by this label. "
                         f"Available columns: {columns}")

    ix = columns.index(x)
    iy = columns.index(y)

    if model.method == "strip":
        retained = set(envelope["keep_columns"])

        if ix not in retained or iy not in retained:
            raise ValueError("A selected column was removed during strip fitting.")
    else:
        retained = set(range(len(columns)))

    def to_nc(frame):
        missing = [c for c in columns if c not in frame.columns]
        if missing:
            raise ValueError(f"Data is missing score columns: {missing}")

        scores = transform_scores(frame[columns].to_numpy(dtype=float), model.score_direction,)

        observed = scores[~np.isnan(scores)]
        if (not np.isfinite(observed).all() or (observed < 0).any()):
            raise ValueError("Observed nonconformity scores must be finite and nonnegative.")
        return scores

    train_nc = to_nc(model.training_points_[label])

    test_frame = load_table(test_data)
    if test_frame.empty:
        raise ValueError("test_data contains no rows.")

    test_nc = to_nc(test_frame)

    # Fix unplotted coordinates at training medians.
    # This chooses the slice; it does not modify the fitted envelope.
    reference = np.array([np.median(column[np.isfinite(column)]) if np.isfinite(column).any() else np.nan
                          for column in train_nc.T])

    if any(not np.isfinite(reference[j]) for j in retained):
        raise ValueError("Cannot define the median slice: an active column has no finite training values.")

    # Include test coordinates so outside points remain visible.
    display_nc = np.concatenate((train_nc, test_nc), axis=0)

    def upper_bound(values):
        finite = values[np.isfinite(values)]
        maximum = float(finite.max())
        return maximum * 1.08 if maximum > 0 else 0.1

    xhi = upper_bound(display_nc[:, ix])
    yhi = upper_bound(display_nc[:, iy])

    gx = np.linspace(0.0, xhi, grid_size)
    gy = np.linspace(0.0, yhi, grid_size)
    XX, YY = np.meshgrid(gx, gy)

    # Full-dimensional grid: only the selected coordinates vary.
    grid_nc = np.tile(reference, (XX.size, 1))
    grid_nc[:, ix] = XX.ravel()
    grid_nc[:, iy] = YY.ravel()

    membership = {"collapsed": collapsed_is_in_region, "radial": radial_is_in_region, 
                  "strip": strip_is_in_region,}[model.method]

    grid_inside = membership(grid_nc, envelope)
    ZZ = grid_inside.reshape(XX.shape).astype(float)

    # Actual full-model candidate decisions.
    # No forced-nonempty fallback is used.
    _, test_inside = model._class_tau_and_membership(test_frame, label)
    test_inside = np.asarray(test_inside, dtype=bool)

    # True labels are optional and affect only point categories.
    test_truth = None

    if model.label_col_ in test_frame.columns:
        if test_frame[model.label_col_].isna().any():
            raise ValueError("Test labels contain missing values. Supply complete "
                             "labels or omit the label column.")

        test_truth = test_frame[model.label_col_].to_numpy()

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    if grid_inside.any():
        ax.contourf(XX, YY, ZZ, levels=[0.5, 1.5], colors=["#b8dfba"], alpha=0.4, zorder=0,)

    if grid_inside.any() and not grid_inside.all():
        ax.contour(XX, YY, ZZ, levels=[0.5], colors=["#482878"], linewidths=2, zorder=2,)
    else:
        ax.text(0.02, 0.98, ("Entire slice grid accepted" if grid_inside.all()
                              else "Entire slice grid rejected"), transform=ax.transAxes,
                va="top", fontsize=8, bbox=dict(facecolor="white", alpha=0.85, edgecolor="none",),)

    def draw_points(points, name, color, marker, alpha):
        px = points[:, ix]
        py = points[:, iy]

        complete = np.isfinite(px) & np.isfinite(py)

        if complete.any():
            ax.scatter(px[complete], py[complete], s=38, color=color, marker=marker, alpha=alpha, 
                       label=f"{name} ({complete.sum()})", zorder=4,)

        if show_missing:
            x_only = np.isfinite(px) & np.isnan(py)
            y_only = np.isnan(px) & np.isfinite(py)

            for value in px[x_only]:
                ax.axvline(value, color=color, linestyle=":", linewidth=1, alpha=0.35,
                           label="_nolegend_", zorder=1)

            for value in py[y_only]:
                ax.axhline(value, color=color, linestyle=":", linewidth=1, alpha=0.35,
                           label="_nolegend_", zorder=1)

    if show_training:
        draw_points(train_nc, f"Training: {label}", "gray", ".", 0.3,)

    if test_truth is None:
        categories = [(test_inside, "Test: accepted", "tab:green", "o",), 
                      (~test_inside, "Test: rejected", "tab:red", "x",),]
    else:
        own_class = test_truth == label

        categories = [(test_inside & own_class, "Test: true accept", "tab:green", "o",),
                      (~test_inside & own_class, "Test: false reject", "tab:red", "x",),
                      (test_inside & ~own_class, "Test: false accept", "tab:orange", "^",),
                      (~test_inside & ~own_class, "Test: true reject", "tab:blue", "x",),]

    for mask, name, color, marker in categories:
        draw_points(test_nc[mask], name, color, marker, 0.8)

    pair = test_nc[:, [ix, iy]]
    complete = np.isfinite(pair).all(axis=1)
    both_missing = np.isnan(pair).all(axis=1)
    partial = ~complete & ~both_missing

    notes = []
    if notes:
        ax.text(0.02, 0.02, "\n".join(notes), transform=ax.transAxes, va="bottom", fontsize=8,
                 bbox=dict(facecolor="white", alpha=0.85, edgecolor="none",), zorder=5,)

    region = Patch(facecolor="#b8dfba", alpha=0.4, label="Accepted region on median slice",)

    handles, labels = ax.get_legend_handles_labels()

    has_missing_lines = any(line.get_linestyle() == ":" for line in ax.lines)

    if has_missing_lines:
        handles.append(Line2D([0], [0], color="gray", linestyle=":", linewidth=1.5,))
        labels.append("One score missing")

    ax.legend([region] + handles, [region.get_label()] + labels, title="Test categories: full-model decisions",
               fontsize=8, title_fontsize=8, loc="best",)

    ax.set(xlabel=x, ylabel=y, title=f"{model.method.capitalize()}: slice for {label}",
            xlim=(0.0, xhi), ylim=(0.0, yhi),)

    return ax


##############################################################################################################

def plot_inclusion_heatmap(prediction_sets, true_labels, *, classes=None, ax=None, 
                           title="Class inclusion rates", annotate=False,):
    """
       Plot how often each candidate appears in prediction sets, grouped by the true class.

       Inputs must have matching sample order.
       Returns (ax, rates), where rates is a pandas DataFrame.

       Classes without evaluation samples have grey rows.
    """
    import matplotlib.pyplot as plt

    prediction_sets = list(prediction_sets)
    true_labels = list(true_labels)

    if not true_labels:
        raise ValueError("No evaluation samples were supplied.")

    if len(prediction_sets) != len(true_labels):
        raise ValueError("prediction_sets and true_labels must have equal lengths.")

    sets = []
    for labels in prediction_sets:
        if isinstance(labels, (str, bytes)):
            raise ValueError("Each prediction set must be a collection of labels, not a string.")
        sets.append(set(labels))

    observed_classes = set(true_labels)
    for labels in sets:
        observed_classes.update(labels)

    if classes is None:
        classes = sorted(observed_classes, key=str)
    else:
        classes = list(classes)

        if not classes or len(set(classes)) != len(classes):
            raise ValueError("classes must be nonempty and unique.")

        unknown = observed_classes - set(classes)
        if unknown:
            raise ValueError(f"classes is missing labels: {sorted(unknown, key=str)}")

    class_index = {label: index for index, label in enumerate(classes)}
    n_classes = len(classes)

    counts = np.zeros((n_classes, n_classes), dtype=float)
    sample_counts = np.zeros(n_classes, dtype=int)

    for true_label, labels in zip(true_labels, sets):
        row = class_index[true_label]
        sample_counts[row] += 1

        for candidate in labels:
            counts[row, class_index[candidate]] += 1

    values = np.full_like(counts, np.nan)
    np.divide(counts, sample_counts[:, None], out=values, where=sample_counts[:, None] > 0,)

    rates = pd.DataFrame(values, index=pd.Index(classes, name="True class"),
                         columns=pd.Index(classes, name="Candidate class"),)

    if ax is None:
        width = max(6, min(20, 0.35 * n_classes + 3))
        height = max(5, min(18, 0.30 * n_classes + 2))
        _, ax = plt.subplots(figsize=(width, height))

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("lightgrey")

    heatmap = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, vmin=0, vmax=1,
                         interpolation="nearest", aspect="auto",)

    positions = np.arange(n_classes)
    ax.set_xticks(positions)
    ax.set_yticks(positions)

    ax.set_xticklabels([str(label) for label in classes], rotation=90,)
    ax.set_yticklabels([f"{label} (n={sample_counts[i]})" for i, label in enumerate(classes)])

    ax.set_xlabel("Candidate class")
    ax.set_ylabel("True class")
    ax.set_title(title)

    colorbar = ax.figure.colorbar(heatmap, ax=ax, pad=0.02)
    colorbar.set_label("Fraction including candidate")

    if annotate:
        for row in range(n_classes):
            for col in range(n_classes):
                value = values[row, col]
                if np.isfinite(value):
                    ax.text(col, row, f"{value:.2f}", ha="center", va="center",
                            color="white" if value < 0.5 else "black",)

    return ax, rates