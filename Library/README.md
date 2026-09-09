# Conformal Envelopes — v0.2

This is a reusable Python-library version of the **collapsed, radial and strip
conformal envelope methods** developed in the supplied cleaned host-classification
notebook.

The public interface follows the professor's handwritten workflow:

```text
TRAINING CSV
ID | s1 | s2 | ... | sk | label
             |
             v
     ConformalSetModel
     alpha + method + parameters
             |
             v
     one envelope per label
```

and then:

```text
TEST CSV + fitted envelopes
             |
             v
     conformal set for each ID
```

The package is deliberately independent of phages, hosts, proteins or biology.
It only sees IDs, numerical predictor scores, labels and missing values.

---

## Folder contents

```text
conformal_envelopes_library_v0_2/
├── conformal_envelopes/
│   ├── __init__.py
│   ├── model.py
│   ├── calibration.py
│   ├── missingness.py
│   ├── io.py
│   ├── metrics.py
│   └── envelopes/
│       ├── collapsed.py
│       ├── radial.py
│       └── strip.py
├── examples/
│   ├── quickstart.py
│   └── make_example_csvs.py
├── tests/
│   └── test_library.py
├── docs/
│   ├── API_GUIDE.md
│   ├── METHOD_MAPPING.md
│   └── PROFESSOR_WORKFLOW.md
├── pyproject.toml
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Installation

Open a terminal in this folder and run:

```bash
pip install -e .
```

For development/testing:

```bash
pip install -e ".[dev]"
```

---

## Expected training CSV

```csv
ID,score_1,score_2,score_3,label
x001,0.91,0.78,,A
x002,0.82,0.74,0.60,A
x003,0.31,0.22,0.44,B
x004,0.25,,0.38,B
```

The first column does **not** technically have to be named `ID`, but `ID` is the
default. Likewise, `label` is the default label column.

Missing predictor values may be ordinary CSV blanks/NaNs.

---

## Minimal example

```python
from conformal_envelopes import ConformalSetModel

model = ConformalSetModel(
    method="radial",
    alpha=0.10,
    n_directions=250,
    smoothing=8.0,
    angle_deg=30.0,
    random_state=13,
)

model.fit(
    "train.csv",
    id_col="ID",
    label_col="label",
)

predictions = model.predict("test.csv")
print(predictions)
```

Output:

```text
ID    prediction_set    set_size    forced
x101  [A]               1           False
x102  [A, B]            2           False
x103  [B]               1           False
```

The full DataFrame also contains `tau_per_label` for inspection.

---

## Dictionary output

```python
prediction_dict = model.predict(
    "test.csv",
    output="dict",
)
```

Example:

```python
{
    "x101": ["A"],
    "x102": ["A", "B"],
    "x103": ["B"],
}
```

---

## Access the learned envelopes

Training produces one envelope for each label:

```python
envelopes = model.get_envelopes()
```

The structure is:

```python
{
    "A": {
        "label": "A",
        "method": "radial",
        "columns": [...],
        "alpha": 0.1,
        "shape_size": ...,
        "calibration_size": ...,
        "parameters": {...},
        "envelope": {...},
    },
    "B": {...},
}
```

So the envelope objects drawn in the professor's training diagram are explicitly
available after `.fit()`.

---

## Choose the envelope method

### Collapsed

```python
model = ConformalSetModel(
    method="collapsed",
    alpha=0.10,
)
```

This computes the NaN-aware mean nonconformity over the available score
dimensions and calibrates a one-dimensional threshold.

### Radial

```python
model = ConformalSetModel(
    method="radial",
    alpha=0.10,
    n_directions=250,
    smoothing=8.0,
    angle_deg=30.0,
    neighbor_fraction=0.2,
)
```

Parameters:

- `n_directions`: number of sampled positive-orthant directions.
- `smoothing`: notebook `kappa`; larger means a sharper/localer angular blend.
- `angle_deg`: angular neighborhood bandwidth for local radial quantiles.
- `neighbor_fraction`: controls the minimum local-support target.

### Strip

```python
model = ConformalSetModel(
    method="strip",
    alpha=0.10,
    n_bins=8,
    smoothing_window=2,
    min_samples=3,
)
```

Parameters:

- `n_bins`: number of bins per conditioning score.
- `smoothing_window`: local forward monotonic smoothing window.
- `min_samples`: minimum local support before using a pure conditional quantile.

---

## Score direction / nonconformity

The original host notebook uses:

```text
nonconformity = 1 - raw score
```

because larger raw scores mean stronger compatibility.

That is the default:

```python
score_direction="higher_is_better"
```

and requires finite raw scores in `[0, 1]`.

If your input is already a nonconformity score where smaller is better:

```python
model = ConformalSetModel(
    method="radial",
    score_direction="lower_is_better",
)
```

Advanced users can supply a callable transformation.

---

## NaN handling

NaNs are preserved instead of being replaced with zero.

The exact handling depends on the envelope geometry and follows the supplied
research implementation.

You can inspect the explicit missingness mask:

```python
mask = model.missingness_mask("test.csv")
```

where:

```text
1 = observed
0 = missing
```

---

## Save the trained envelopes

```python
model.save("my_conformal_model.pkl")
```

Later:

```python
from conformal_envelopes import ConformalSetModel

model = ConformalSetModel.load(
    "my_conformal_model.pkl"
)

prediction_sets = model.predict("new_test.csv")
```

---

## 2D visualization

The professor's second sketch asks for two selected score columns and a
two-dimensional envelope view.

```python
model.plot(
    x="score_1",
    y="score_2",
    label="A",
)
```

For a model fitted in more than two dimensions, the other dimensions are treated
as missing for the visualization, so the same NaN-aware rules are used.

---

## Advanced: label-specific score columns

This is **not needed for the generic professor-style CSV**.

It is retained because the original biological host application had
candidate-label-specific score groups.

```python
mapping = {
    "A": ["A_s1", "A_s2"],
    "B": ["B_s1", "B_s2"],
}

model.fit(
    train_df,
    score_cols=["A_s1", "A_s2", "B_s1", "B_s2"],
    label_to_columns=mapping,
)
```

---

## Run tests

```bash
pytest -q
```

The tests exercise all three envelope methods, NaN handling, prediction-set
outputs, save/load, and explicit envelope access.

---

## What this version intentionally does not do yet

This is a research-library prototype rather than a mature PyPI release.

Before public release, you would normally add:

- continuous integration (GitHub Actions),
- a versioned changelog,
- more regression tests against saved notebook outputs,
- API documentation generation,
- benchmarking,
- a formal citation/reference section for the underlying method.

The mathematical core is separated from the original application so these can
be added without redesigning the package.
