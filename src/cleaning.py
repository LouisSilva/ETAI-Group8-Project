"""Fixes errors in the raw data before any modelling.

`clean_dataset` is the only function the rest of the project calls (main.py, predict.py). It runs
the cleaning steps in a fixed order: rename columns -> correct types -> placeholder values ->
sign errors -> domain rules.

Each step has its own function, so it can also be used on its own in the EDA notebook. They all
follow the same rules:
- Name: `apply_<what it does>(df, <its settings>)`.
- They change `df` in place. Only `clean_dataset` makes a copy, so the raw data is never modified.
- Their settings come from the `cleaning:` section of config.yaml. An empty setting means the step
  does nothing.
- They work row by row (no information from other rows), so there is no leakage and the same
  cleaning can be applied to train.csv, test.csv or a single purchase. No rows are removed.
"""
import numpy as np
import pandas as pd


def apply_column_renames(df: pd.DataFrame, rename_columns: dict) -> None:
    """
    Renames columns via the {"old column name": "new column name"} mapping, in place.
    Columns not in the mapping keep their name.
    """
    df.rename(columns=rename_columns, inplace=True)


def apply_correct_types(df: pd.DataFrame, column_types: dict, date_format: str) -> None:
    """
    Converts all observations in a column to a specified type:
    
        - Columns in `int_columns` get converted to the pandas Int64 type.
        - Columns in `date_columns` get converted to the date type using the specified date format.
    """
    for col in column_types.get("int", []):
        if col not in df.columns: continue
        df[col] = (pd.to_numeric(df[col], errors="coerce")).astype("Int64")

    for col in column_types.get("date", []):
        if col not in df.columns: continue
        
        stripped_values = df[col].astype("string").str.strip()
        df[col] = pd.to_datetime(stripped_values, format=date_format, errors="coerce")


def apply_placeholder_values(df: pd.DataFrame, placeholder_values: dict) -> pd.DataFrame:
    """
    Replaces placeholder values {column: [values]} (codes that mean "unknown", e.g. 1e9) by NaN,
    in place. Returns the number of values replaced per column.
    """
    report_rows = []
    for col, values in placeholder_values.items():
        if col not in df.columns:
            continue

        is_placeholder = df[col].isin(values)
        report_rows.append({"column": col, "placeholder": values, "replaced": int(is_placeholder.sum())})
        df.loc[is_placeholder, col] = np.nan

    return pd.DataFrame(report_rows)



def apply_sign_fixes(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Replaces negative values by their absolute value, in place, for columns where a negative value
    is the right number with a wrong sign. Returns the number of values fixed per column.
    """
    report_rows = []
    for col in columns:
        if col not in df.columns:
            continue

        is_negative = df[col] < 0
        report_rows.append({"column": col, "fixed": int(is_negative.sum())})
        df.loc[is_negative, col] = df.loc[is_negative, col].abs()

    return pd.DataFrame(report_rows)


def apply_domain_rules(df: pd.DataFrame, rules: dict) -> pd.DataFrame:
    """
    Applies {column: {"min": ..., "max": ...}} domain rules (either bound optional) and sets
    violations to NaN, in place. Returns the number of violations per column.
    """
    report_rows = []
    for col, bounds in rules.items():
        if col not in df.columns:
            continue
        
        numeric = pd.to_numeric(df[col], errors="coerce")
        lower_ok = numeric >= bounds["min"] if "min" in bounds else pd.Series(True, index=numeric.index)
        upper_ok = numeric <= bounds["max"] if "max" in bounds else pd.Series(True, index=numeric.index)
        violations = numeric.notna() & ~(lower_ok & upper_ok)
        
        report_rows.append({"column": col, "rule": bounds, "violations": int(violations.sum())})
        df.loc[violations, col] = np.nan
        
    return pd.DataFrame(report_rows)


def clean_dataset(df: pd.DataFrame, cleaning_config: dict) -> pd.DataFrame:
    """Applies the cleaning rules in `cleaning_config`"""
    out = df.copy()

    # Rename columns using the configured mappings
    apply_column_renames(out, cleaning_config.get("rename_columns", {}))

    # Converts columns to their correct types
    apply_correct_types(out, cleaning_config.get("column_types", {}), cleaning_config["date_format"])

    # Replace placeholder values (codes meaning "unknown", e.g. 1e9) by NaNs
    apply_placeholder_values(out, cleaning_config.get("placeholder_values", {}))

    # Fix sign errors
    apply_sign_fixes(out, cleaning_config.get("sign_flip_columns", []))

    # Apply the {column: {"min": ..., "max": ...}} domain rules
    apply_domain_rules(out, cleaning_config.get("domain_rules", {}))

    return out