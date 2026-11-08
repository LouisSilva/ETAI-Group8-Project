from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline
# import joblib

from .data import load_data
from .cleaning import clean_dataset


def create_kaggle_submission(
    pipeline: Pipeline,
    X_full: pd.DataFrame,
    config: dict
) -> None:
    """
    Trains the final pipeline on the full dataset, predicts the labels in the test set, and creates a kaggle submission csv.
    """
    df_kaggle = clean_dataset(
        load_data(config["data"]["csv_paths"]["test"]),
        config["cleaning"]["train"]
    )

    drop_columns = set(config["preprocessing"].get("drop_columns", []))
    X_kaggle = df_kaggle.drop(columns=list(drop_columns), errors="ignore").copy()
    X_kaggle = X_kaggle[X_full.columns]

    predictions = pipeline.predict(X_kaggle).astype(int)
    submission = pd.DataFrame({
        "ID": df_kaggle["ID"].to_numpy(),
        "repeat_purchase_90d": predictions
    })

    # Sanity checks
    assert len(submission) == 986
    assert submission["ID"].is_unique
    assert submission["repeat_purchase_90d"].isin([0, 1]).all()

    Path("results").mkdir(exist_ok=True)
    submission.to_csv("results/submission.csv", index=False)

    # Path("models").mkdir(exist_ok=True)
    # joblib.dump(pipeline, "models/classification.joblib")