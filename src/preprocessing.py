import math
from typing import Any

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, TargetEncoder, StandardScaler, MinMaxScaler, RobustScaler
from category_encoders import CountEncoder


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
    y_classification = df.get(classification_target)
    y_regression = df.get(regression_target)

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


_SCALERS: dict[str, Any] = {
    "none": "passthrough",
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler
}

_ENCODERS: dict[str, Any] = {
    "onehot": lambda seed: OneHotEncoder(handle_unknown="ignore", sparse_output=False),
    "ordinal": lambda seed: OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
    "count": lambda seed: CountEncoder(handle_unknown=0, handle_missing=0),
    "target": lambda seed: TargetEncoder(target_type="binary", cv=StratifiedKFold(5, shuffle=True, random_state=seed)),
}

# Numeric imputers. "iterative" is MICE with one completed dataset; "random_forest" is the
# missForest idea (each column predicted from the others by a random forest). Both learn only
# from the feature columns, never from the target, so they also run on data to predict.
_NUMERIC_IMPUTERS: dict[str, Any] = {
    "simple_imputer": lambda cfg, seed: SimpleImputer(
        strategy=cfg.get("strategy", "median"),
        add_indicator=cfg.get("add_indicator", False)
    ),

    "knn": lambda cfg, seed: KNNImputer(
        n_neighbors=cfg.get("n_neighbors", 5),
        add_indicator=cfg.get("add_indicator", False)
    ),

    "iterative": lambda cfg, seed: IterativeImputer(
        max_iter=cfg.get("max_iter", 10),
        add_indicator=cfg.get("add_indicator", False),
        random_state=seed
    ),

    "random_forest": lambda cfg, seed: IterativeImputer(
        estimator=RandomForestRegressor(
            n_estimators=cfg.get("n_estimators", 50),
            min_samples_leaf=cfg.get("min_samples_leaf", 5),
            random_state=seed,
            n_jobs=cfg.get("n_jobs", 1)),
        max_iter=cfg.get("max_iter", 10),
        n_nearest_features=cfg.get("n_nearest_features", None),
        add_indicator=cfg.get("add_indicator", False),
        random_state=seed
    ),
}

_CATEGORICAL_IMPUTERS: dict[str, Any] = {
    "simple_imputer": lambda cfg, seed: SimpleImputer(
        strategy=cfg.get("strategy", "most_frequent"),
        add_indicator=cfg.get("add_indicator", False)
    ),
}


def build_preprocessor(preprocessing_config: dict) -> ColumnTransformer:
    """
    Builds the unfitted ColumnTransformer described by `preprocessing_config`: scaling and 
    imputation for numeric columns (scaled first, so distance-based imputers compare columns
    on one scale; scalers ignore NaN when fitting), imputation and encoding for categorical 
    columns, and the missingness flags passed through. Every encoder tolerates unseen categories.
    """
    scaler_factory = _SCALERS[preprocessing_config["scaler"]]
    scaler = scaler_factory() if callable(scaler_factory) else scaler_factory
    encoder = _ENCODERS[preprocessing_config["encoder"]](preprocessing_config.get("random_state"))

    selected_numeric_imputer = preprocessing_config["numeric_imputation"].get("selected_imputer", "simple_imputer")
    if selected_numeric_imputer not in _NUMERIC_IMPUTERS:
        raise ValueError(f"Unknown numeric_strategy: {selected_numeric_imputer}. Options: {list(_NUMERIC_IMPUTERS)}")

    numeric_pipeline = Pipeline([
        ("scale", scaler),
        ("impute", _NUMERIC_IMPUTERS[selected_numeric_imputer](preprocessing_config["numeric_imputation"].get(selected_numeric_imputer, {}), preprocessing_config.get("random_state"))),
    ])

    selected_categorical_imputer = preprocessing_config["categorical_imputation"].get("selected_imputer", "simple_imputer")
    if selected_categorical_imputer not in _CATEGORICAL_IMPUTERS:
        raise ValueError(f"Unknown numeric_strategy: {selected_categorical_imputer}. Options: {list(_CATEGORICAL_IMPUTERS)}")
    
    categorical_pipeline = Pipeline([
        ("impute", _CATEGORICAL_IMPUTERS[selected_categorical_imputer](preprocessing_config["categorical_imputation"].get(selected_categorical_imputer, {}), preprocessing_config.get("random_state"))),
        ("encode", encoder),
    ])
    
    return ColumnTransformer([
        ("numeric", numeric_pipeline, preprocessing_config["numeric_features"]),
        ("categorical", categorical_pipeline, preprocessing_config["categorical_features"]),
    ])
