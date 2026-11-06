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


def train_validation_split(
    X: pd.DataFrame, 
    y_classification: pd.Series, 
    y_regression: pd.Series,
    split_config: dict
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Returns (X_train, X_validation, y_classification_train, y_classification_validation, y_regression_train, y_regression_validation)
    """
    dates = X["purchase_date"]
    validation_start = pd.Timestamp(split_config["validation_start"])
    validation_end = pd.Timestamp(split_config["validation_end"])
    horizon = pd.Timedelta(days=split_config["horizon_days"])

    # Keep observations with valid classification labels and complete 90-day observation windows
    eligible = y_classification.isin([0, 1]) & dates.notna()

    # Chronological split with a 90-day gap
    train_mask = eligible & (dates + horizon < validation_start)
    validation_mask = eligible & dates.between(validation_start, validation_end)

    return (
        X.loc[train_mask].copy(),
        X.loc[validation_mask].copy(),
        y_classification.loc[train_mask].copy(),
        y_classification.loc[validation_mask].copy(),
        y_regression.loc[train_mask].copy(),
        y_regression.loc[validation_mask].copy(),
    )


_SCALERS = {"none": "passthrough", "standard": StandardScaler, "minmax": MinMaxScaler, "robust": RobustScaler}
_ENCODERS = {
    "onehot": lambda seed: OneHotEncoder(handle_unknown="ignore", sparse_output=False),
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