# Professor workflow mapped to the library

## Training

Handwritten idea:

```text
CSV:
ID | s1 | ... | sk | label
             |
             v
           library
     alpha + envelope method
     + method parameters
             |
             v
        data structure E
        one per class label
```

Library equivalent:

```python
model = ConformalSetModel(
    method="strip",
    alpha=0.1,
    n_bins=8,
)

model.fit("train.csv")

E = model.envelopes_
```

`E` is keyed by label.

## Prediction

Handwritten idea:

```text
test CSV + fitted E
        |
        v
conformal set for each test ID
```

Library equivalent:

```python
sets = model.predict("test.csv")
```

or:

```python
sets = model.predict(
    "test.csv",
    output="dict",
)
```

## Visualization

Handwritten idea:

```text
CSV + envelope + 2 specified columns
                    |
                    v
                 2D plot
```

Library equivalent:

```python
model.plot(
    x="score_1",
    y="score_2",
    label="A",
)
```
