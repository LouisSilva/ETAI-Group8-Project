import numpy as np
import pandas as pd

def clean_dataset(df: pd.DataFrame, cleaning_config: dict) -> pd.DataFrame:
    """Applies the cleaning rules in `cleaning_config`"""
    out = df.copy()

    # Rename columns using the configured mappings
    out = out.rename(columns=cleaning_config.get("rename_columns", {}))

    # Apply the {column: {"min": ..., "max": ...}} domain rules
    apply_domain_rules(out, cleaning_config.get("domain_rules", {}))

    # Converts columns to their correct types
    apply_correct_types(out, cleaning_config.get("column_types", {}), cleaning_config["date_format"])

    return out

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

def apply_correct_types(df: pd.DataFrame, column_types: dict, date_format: str):
    """
    Converts all observations in a column to a specified type:
    
        - Columns in `int_columns` get converted to the pandas Int64 type.
        - Columns in `date_columns` get converted to the date type using the specified date format.
    """
    for col in column_types.get("int", []):
        if col in df.columns:
            df[col] = (pd.to_numeric(df[col], errors="coerce")).astype("Int64")

    for col in column_types.get("date", []):
        if col in df.columns:
            stripped_values = df[col].astype("string").str.strip()
            df[col] = pd.to_datetime(stripped_values, format=date_format, errors="coerce")
        