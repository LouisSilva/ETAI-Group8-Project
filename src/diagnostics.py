"""
Data diagnostics: checks that measure problems in the data, without changing it.

Used in the EDA notebook (notebooks/EDA.ipynb), not by main.py. Each finding becomes a rule in
config.yaml, applied by `clean_dataset` (src/cleaning.py). Every function returns a report and
leaves the data it receives unchanged.
"""
import numpy as np
import pandas as pd


def find_negative_values(df: pd.DataFrame, exclude: list[str] = ()) -> pd.DataFrame:
    """Numeric columns with negative values: count, number of distinct values, min and max."""
    rows = {}
    for col in df.select_dtypes("number").columns:
        if col in exclude:
            continue

        negative_values = df.loc[df[col] < 0, col]
        if negative_values.empty:
            continue

        rows[col] = {
            "negative values": len(negative_values),
            "distinct negative values": negative_values.nunique(),
            "min": negative_values.min(),
            "max": negative_values.max(),
        }

    return pd.DataFrame(rows).T


def check_sign_errors(df: pd.DataFrame, rules: dict) -> pd.DataFrame:
    """
    Counts the negative values that become correct once the sign is removed. Each rule gives the
    true value: {"min_column": c} (at least c) or {"numerator": a, "denominator": b} (exactly a / b).
    """
    rows = []
    for col, rule in rules.items():
        negative = df[col] < 0
        without_sign = df.loc[negative, col].abs().astype(float)

        if "min_column" in rule:
            test = f"|value| >= {rule['min_column']}"
            lower_bound = df.loc[negative, rule["min_column"]].astype(float)
            matches = without_sign >= lower_bound - 0.01   # tolerance for rounding to the cent
        else:
            test = f"|value| == {rule['numerator']} / {rule['denominator']}"
            numerator = df.loc[negative, rule["numerator"]].astype(float)
            denominator = df.loc[negative, rule["denominator"]].astype(float)
            matches = np.isclose(without_sign, numerator / denominator)

        rows.append({
            "column": col,
            "negative values": int(negative.sum()),
            "test": test,
            "match without the sign": int(matches.sum()),
        })

    return pd.DataFrame(rows)


def check_recency_rule(df: pd.DataFrame, days_column: str, count_column: str, window_days: int) -> pd.DataFrame:
    """
    A purchase less than `window_days` ago must be counted in `count_column`. Checks the rule on the
    valid rows, then counts the negative rows it rules out as sign errors.
    """
    days = df[days_column]
    no_recent_purchase = df[count_column] == 0

    within_window = days.between(0, window_days - 1)
    negative = days < 0
    sign_flip_impossible = negative & (days.abs() < window_days) & no_recent_purchase

    return pd.DataFrame(
        {
            "rows": [int(within_window.sum()), int(negative.sum())],
            f"{count_column} = 0 (contradiction)": [
                int((within_window & no_recent_purchase).sum()),
                int(sign_flip_impossible.sum()),
            ],
        },
        index=[
            f"valid rows, {days_column} < {window_days}",
            f"negative rows, |{days_column}| < {window_days}",
        ],
    )