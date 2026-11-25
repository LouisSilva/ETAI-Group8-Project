from pathlib import Path
import joblib

import pandas as pd
import numpy as np

from .config import load_yaml_config
from .cleaning import clean_dataset
from .preprocessing import split_features_target

def predict(
    input_df: pd.DataFrame,
    config: dict,
    classifier,
    regressor=None,
    output_probability: bool = False,
    output_prediction: bool = True, # The default settings are to have just ID and the prediction, which is for the kaggle submission
    output_regression: bool = False
) -> pd.DataFrame:
    """Predict requested outputs, keeping one row per input observation."""
    if not any([output_probability, output_prediction, output_regression]):
        raise ValueError("At least one output flag must be enabled.")

    if output_regression and regressor is None:
        raise ValueError("A regressor is required for regression predictions.")

    cleaned_df = clean_dataset(input_df, config["cleaning"]["train"])

    X, _, _ = split_features_target(cleaned_df, config["preprocessing"])
    X_clf = X[classifier.feature_names_in_]

    predictions = pd.DataFrame({"ID": cleaned_df["ID"].to_numpy()})

    if output_probability:
        predictions["proba__repeat_purchase_90d"] = classifier.predict_proba(X_clf)[:, 1]

    # Return early if the other two options aren't enabled
    if not output_prediction and not output_regression: return predictions

    y_clf_pred = classifier.predict(X_clf).astype(int)
    if output_prediction:
        predictions["repeat_purchase_90d"] = y_clf_pred

    if output_regression:
        returning_customers = y_clf_pred == 1
        y_reg_pred = np.full(len(X), np.nan)

        if returning_customers.any():
            X_reg = X[regressor.feature_names_in_]
            y_reg_pred[returning_customers] = regressor.predict(X_reg.iloc[np.flatnonzero(returning_customers)])

        predictions["spend_next_90d"] = y_reg_pred

    return predictions


def predict_csv(
    classifier_path: Path,
    regressor_path: Path | None = None,
    input_path: Path | None = None,
    output_path: Path | None = None,
    output_probability: bool = False,
    output_prediction: bool = True,
    output_regression: bool = False,
) -> None:
    """Loads saved models, predict a CSV, and save the requested outputs."""
    config = load_yaml_config()

    input_path = Path(input_path or config["data"]["csv_paths"]["test"])
    output_path = Path(output_path or "results/predictions.csv")

    classifier = joblib.load(classifier_path)
    regressor = joblib.load(regressor_path) if output_regression else None

    predictions = predict(
        input_df=pd.read_csv(input_path),
        config=config,
        classifier=classifier,
        regressor=regressor,
        output_probability=output_probability,
        output_prediction=output_prediction,
        output_regression=output_regression,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)

    print(f"{len(predictions)} rows predicted -> {output_path}")
