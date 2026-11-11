from datetime import datetime

import pandas as pd
import numpy as np

from sklearn.base import clone

from src.config import load_yaml_config
from src.data import load_data
from src.cleaning import clean_dataset
from src.preprocessing import split_features_target, development_test_split
from src.experiments import run_classification_experiment
from src.results import save_run, save_scores, build_scoreboard
from src.kaggle_submission import create_kaggle_submission


def main():
    config = load_yaml_config()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load and clean the dataset
    df_raw = load_data(config["data"]["csv_paths"]["train"])
    df_cleaned = clean_dataset(df_raw, config["cleaning"]["train"])

    # Split the dataframe into features (X), and the targets (y_clf = the classification target, and y_reg = the regression target)
    preprocessing_config = config["preprocessing"]
    X, y_clf, y_reg = split_features_target(df_cleaned, preprocessing_config)

    # Create a development and test set for both the classification and regression task. 
    # Every decision uses the development set only; the test set stays locked
    X_dev, X_test, y_clf_dev, y_clf_test, y_reg_dev, y_reg_test = development_test_split(X, y_clf, y_reg, config["split"])

    preprocessing_config = {
        **config["preprocessing"],
        "numeric_features": X_dev.select_dtypes(include="number").columns.tolist(),
        "categorical_features": X_dev.select_dtypes(include=["object", "category", "string"]).columns.tolist(),
    }
    config = {**config, "preprocessing": preprocessing_config}

    # ----- Classification experiment ----- #
    experiment = run_classification_experiment(X_dev, y_clf_dev, config, run_id)

    # ----- Save report and scoreboard ----- #
    report = experiment.report
    row = report.to_scoreboard_row()
    runs = save_scores(config.get("output", {}).get("scoreboard", "results/runs.csv"), row)
    report.scoreboard = build_scoreboard(runs, row)
    report_path = save_run(config.get("output", {}).get("results_dir", "results"), run_id, config, str(report))

    print(f"CV: {row['mean']:.3f} ± {row['std']:.3f}")
    print(f"Report saved: {report_path}")

    # ----- Fit final classifier on all labelled train.csv observations ----- #
    final_classifier = experiment.selected_pipeline
    labelled = y_clf.isin([0, 1])
    final_classifier.fit(X.loc[labelled], y_clf.loc[labelled].astype(int))

    # ----- Create Kaggle submission ----- #
    create_kaggle_submission(final_classifier, X, config)

    # Future regression stage: 
    # 1). Train and tune the regression pipeline using `X_dev`, restricted to rows where `y_clf_dev == 1` and `y_reg_dev` is known.
    # 2). Use `y_clf_pred = final_classifier.predict(X)` to predict customers that will make another purchase.
    # 3). For those customers that are predicted to return (use the condition `X[y_clf_pred == 1]`), predict how much they will spend in the 90 days after the current purchase.


if __name__ == "__main__":
    main()