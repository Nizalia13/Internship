from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_table(data):
    """Load a pandas DataFrame or CSV path."""
    if isinstance(data, pd.DataFrame):
        return data.copy()

    path = Path(data)
    if path.suffix.lower() != ".csv":
        raise ValueError("Input must be a pandas DataFrame or a .csv file.")
    return pd.read_csv(path)


def infer_score_columns(df, id_col: str, label_col: str | None):
    """Infer numeric predictor columns after removing ID and optional label."""
    excluded = {id_col}
    if label_col is not None and label_col in df.columns:
        excluded.add(label_col)

    columns = [c for c in df.columns if c not in excluded]

    if not columns:
        raise ValueError("No predictor score columns were found.")

    nonnumeric = [
        c for c in columns
        if not pd.api.types.is_numeric_dtype(df[c])
    ]
    if nonnumeric:
        raise ValueError(
            "Inferred predictor columns must be numeric. "
            f"Non-numeric columns: {nonnumeric}. "
            "Pass score_cols explicitly if the CSV has metadata columns."
        )

    return columns
