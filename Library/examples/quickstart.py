from pathlib import Path

from conformal_envelopes import (
    ConformalSetModel,
    evaluate_prediction_sets,
)


here = Path(__file__).resolve().parent

train_csv = here / "train_example.csv"
test_csv = here / "test_example.csv"

model = ConformalSetModel(
    method="radial",
    alpha=0.10,
    n_directions=100,
    smoothing=8.0,
    angle_deg=30.0,
    random_state=13,
)

model.fit(train_csv)

predictions = model.predict(test_csv)

print(
    predictions[
        ["ID", "prediction_set", "set_size", "forced"]
    ].head()
)

# Since the example test CSV contains labels, we can evaluate it.
import pandas as pd

test = pd.read_csv(test_csv)

metrics = evaluate_prediction_sets(
    predictions["prediction_set"],
    test["label"],
)

print(metrics)

# The learned object explicitly contains one envelope per label.
print(model.get_envelopes().keys())

# Optional visualization:
# model.plot(x="score_1", y="score_2", label="A")
