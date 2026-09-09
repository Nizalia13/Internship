# API guide

## Main class

```python
from conformal_envelopes import ConformalSetModel
```

### Constructor

```python
ConformalSetModel(
    method="collapsed",
    alpha=0.10,
    score_direction="higher_is_better",
    shape_fraction=0.5,
    force_nonempty=True,
    random_state=13,
    min_class_size=4,
    **method_params,
)
```

## Training

```python
model.fit(
    data,
    id_col="ID",
    label_col="label",
    score_cols=None,
)
```

`data` may be a CSV path or pandas DataFrame.

If `score_cols=None`, every numeric column except ID and label is treated as a
predictor score.

## Prediction

```python
model.predict(
    data,
    output="dataframe",
)
```

Output choices:

- `"dataframe"`
- `"dict"`
- `"records"`

## Learned envelope structure

```python
model.get_envelopes()
```

## Missingness mask

```python
model.missingness_mask(data)
```

## Visualization

```python
model.plot(
    x="score_1",
    y="score_2",
    label="A",
)
```

## Persistence

```python
model.save("model.pkl")
model = ConformalSetModel.load("model.pkl")
```
