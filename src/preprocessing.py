import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler


def split_features_target(df: pd.DataFrame, data_config: dict) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Returns (X, y_classification, y_regression)
    """
    classification_target = data_config["classification_target"]
    regression_target = data_config["regression_target"]
    
    drop_columns = set(data_config.get("drop_columns", []))
    drop_columns.update([classification_target, regression_target])

    X = df.drop(columns=list(drop_columns), errors="ignore").copy()
    y_classification = df[classification_target].copy()
    y_regression = df[regression_target].copy()

    return X, y_classification, y_regression


def train_validation_split(
    X: pd.DataFrame, 
    y_classification: pd.Series, 
    y_regression: pd.Series,
    split_config: dict
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Returns (X_train, X_validation, y_classification_train, y_classification_validation, y_regression_train, y_regression_validation)
    """
    dates = df["purchase_date"]
    start = pd.Timestamp(split_config["validation_start"])
    end = pd.Timestamp(split_config["validation_end"])
    horizon = pd.Timedelta(days=split_config["horizon_days"])
    observation_end = pd.Timestamp(dates.max())

    # Keep observations with valid classification labels and complete 90-day observation windows
    eligible = y_classification.isin([0, 1])
    eligible &= dates.notna() & (dates + horizon <= observation_end)

    # Chronological split with a 90-day gap
    train_mask = eligible & (dates + horizon < start)
    validation_mask = eligible & dates.between(start, end)

    return (
        X.loc[train_mask].copy(),
        X.loc[validation_mask].copy(),
        y_classification.loc[train_mask].copy(),
        y_classification.loc[validation_mask].copy(),
        y_regression.loc[train_mask].copy(),
        y_regression.loc[validation_mask].copy(),
    )
