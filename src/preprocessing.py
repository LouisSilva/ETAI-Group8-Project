import math
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, TargetEncoder, StandardScaler, MinMaxScaler, RobustScaler


def split_features_target(df: pd.DataFrame, preprocessing_config: dict) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Takes the DataFrame as input and splits it into the feature set (X), and the target sets for the classification and regression problem.
    
    Returns (X, y_classification, y_regression)
    """
    classification_target = preprocessing_config["classification_target"]
    regression_target = preprocessing_config["regression_target"]
    
    drop_columns = set(preprocessing_config.get("drop_columns", []))
    drop_columns.update([classification_target, regression_target])

    X = df.drop(columns=list(drop_columns), errors="ignore").copy()
    y_classification = df[classification_target].copy()
    y_regression = df[regression_target].copy()

    return X, y_classification, y_regression


def development_test_split(
    X: pd.DataFrame, 
    y_classification: pd.Series, 
    y_regression: pd.Series,
    split_config: dict
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Returns (X_dev, X_test, y_clf_dev, y_clf_test, y_reg_dev, y_reg_test), where clf = classification, and reg = regression.
    """
    dates = X["purchase_date"]
    horizon_length = pd.Timedelta(days=split_config["horizon_length_days"])
    test_size = split_config["test_size"]

    if not 0 < test_size < 1:
        raise ValueError(f"test_size must be between 0 and 1. Currently, it is set to {test_size}.")

    # Keep observations with valid classification labels and dates
    eligible = y_classification.isin([0, 1]) & dates.notna()
    eligible_dates = dates.loc[eligible].sort_values(ascending=True)

    if len(eligible_dates) < 2:
        raise ValueError("Not enough eligible observations to split")

    # Number of observations to use for the test set
    n_test = math.ceil(len(eligible_dates) * test_size)

    # First date of the test period
    test_start_date = eligible_dates.iloc[-n_test].normalize()

    dev_mask = eligible & (dates + horizon_length < test_start_date)
    test_mask = eligible & (dates >= test_start_date)

    return (
        X.loc[dev_mask].copy(),
        X.loc[test_mask].copy(),
        y_classification.loc[dev_mask].copy(),
        y_classification.loc[test_mask].copy(),
        y_regression.loc[dev_mask].copy(),
        y_regression.loc[test_mask].copy(),
    )


_SCALERS = {
    "none": "passthrough",
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler
}
_ENCODERS = {
    "onehot": lambda seed: OneHotEncoder(handle_unknown="ignore", sparse_output=False),
    # "onehot2": lambda seed: OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10, sparse_output=False),
    "ordinal": lambda seed: OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
    "count": lambda seed: CountEncoder(handle_unknown=0, handle_missing=0),
}


def build_preprocessor(preprocessing_config: dict) -> ColumnTransformer:
    """
    Builds the unfitted ColumnTransformer described by `preprocessing_config`: imputation and
    scaling for numeric columns, imputation and encoding for categorical ones, and the
    missingness flags passed through. Every encoder tolerates unseen categories.
    """
    imputation = preprocessing_config.get("imputation", {})
    scaler_factory = _SCALERS[preprocessing_config["scaler"]]
    scaler = scaler_factory() if callable(scaler_factory) else scaler_factory
    encoder = _ENCODERS[preprocessing_config["encoder"]](preprocessing_config.get("random_state"))

    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy=imputation.get("numeric_strategy", "median"))),
        ("scale", scaler),
    ])
    
    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy=imputation.get("categorical_strategy", "most_frequent"))),
        ("encode", encoder),
    ])
    
    indicator_cols = [f"{c}_was_missing" for c in preprocessing_config.get("mnar_indicator_sources", [])]
    
    return ColumnTransformer([
        ("numeric", numeric_pipeline, preprocessing_config["numeric_features"]),
        ("categorical", categorical_pipeline, preprocessing_config["categorical_features"]),
        ("indicators", "passthrough", indicator_cols),
    ])