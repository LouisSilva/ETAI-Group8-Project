import shutil
import tempfile
from datetime import datetime

import pandas as pd
import numpy as np

from sklearn.base import clone

from src.config import load_yaml_config
from src.data import load_data
from src.cleaning import clean_dataset
from src.preprocessing import split_features_target, development_test_split
from src.sklearn_extensions import CalendarHorizonSplit
from src.model import build_pipeline
from src.evaluation import build_scorers, cross_validate_pipeline
from src.tuning import tune_pipeline, nested_cross_validate
from src.reports import ClassificationReport
from src.results import save_run, save_scores, build_scoreboard
from src.kaggle_submission import create_kaggle_submission


def main():
    config = load_yaml_config()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load and clean the dataset
    df_raw = load_data(config["data"]["csv_paths"]["train"])
    df_cleaned = clean_dataset(df_raw, config["cleaning"]["train"])

    # Create a train and validation set for both the classification and regression task. clf = classification, reg = regression
    preprocessing_config = config["preprocessing"]
    X, y_clf, y_reg = split_features_target(df_cleaned, preprocessing_config)

    # Every decision uses the development set only; the test set stays locked
    X_dev, X_test, y_clf_dev, y_clf_test, y_reg_dev, y_reg_test = development_test_split(X, y_clf, y_reg, config["split"])

    preprocessing_config = {
        **config["preprocessing"],
        "numeric_features": X_dev.select_dtypes(include="number").columns.tolist(),
        "categorical_features": X_dev.select_dtypes(include=["object", "category", "string"]).columns.tolist(),
    }

    with tempfile.TemporaryDirectory(prefix="pipeline_cache_") as cache_dir:
    
        pipeline = build_pipeline(preprocessing_config, config["models"], config["feature_selection"], memory=cache_dir)
        pipeline.verbose = config.get("pipeline_verbose_enabled", False)
    
        # ----- Cross-Validate Selected Model ----- #

        selected_model = config["models"]["selected_model"]
        cv_config = config["cv"]
        selection_metric = cv_config["selection_metric"]
        metrics = cv_config["metrics"]
        scorers = build_scorers(metrics)
        n_jobs = cv_config.get("n_jobs", 1)
        
        outer_cv = CalendarHorizonSplit(
            date_column="purchase_date",
            horizon_length_days=cv_config["horizon_length_days"],
            validation_length_days=cv_config["validation_length_days"],
            n_splits=cv_config["outer_splits"]
        )
    
        inner_cv = CalendarHorizonSplit(
            date_column="purchase_date",
            horizon_length_days=cv_config["horizon_length_days"],
            validation_length_days=cv_config["validation_length_days"],
            n_splits=cv_config["inner_splits"]
        )

        floor_pipeline = build_pipeline(preprocessing_config, {"selected_model": "dummy"})
        floor_scores, _ = cross_validate_pipeline(
            floor_pipeline,
            X_dev,
            y_clf_dev,
            outer_cv,
            {selection_metric: scorers[selection_metric]},
            selection_metric,
            n_jobs
        )
        
        untuned_scores, _ = cross_validate_pipeline(
            pipeline,
            X_dev,
            y_clf_dev,
            outer_cv,
            scorers,
            selection_metric,
            n_jobs,
        )
    
        # ----- Nested CV and Hyperparameter Tuning ----- #

        study = None
        nested_scores = None
        oof_true = None
        oof_proba = None
    
        tuning_config = config.get("tuning", {})
        tuning_enabled = tuning_config.get("enabled", False)
        selection_enabled = config["feature_selection"].get("enabled", False)
        if tuning_enabled:
            search_space = dict(tuning_config["search_spaces"].get(selected_model, {}))
            if len(search_space) == 0:
                raise ValueError(f"Tuning is enabled, but the selected model '{selected_model}' has no search spaces defined in config.yaml.")
    
            if selection_enabled:
                search_space.update(config["feature_selection"].get("search_spaces", {}))

            # print(f"\nTuning {tuning_config["n_trials"]} trials, nested CV", flush=True)
            nested_scores, probabilities = nested_cross_validate(
                pipeline,
                X_dev,
                y_clf_dev,
                outer_cv,
                inner_cv,
                scorers,
                selection_metric,
                search_space,
                tuning_config["n_trials"],
                tuning_config["random_state"],
                n_jobs=n_jobs
            )
    
            # Returns a pipeline with selected parameters.
            pipeline, study = tune_pipeline(
                pipeline,
                X_dev,
                y_clf_dev,
                inner_cv,
                scorers[selection_metric],
                search_space,
                tuning_config["n_trials"],
                tuning_config["random_state"],
                n_jobs=n_jobs
            )

            valid = np.isfinite(probabilities)
            oof_true = y_clf_dev.iloc[valid]
            oof_proba = probabilities[valid]

        # ----- Fit Selected Pipeline on Development Data ----- #
        pipeline.fit(X_dev, y_clf_dev)
        
        selected_features = None
        total_features = None
        selected_min_votes = None
        
        if selection_enabled:
            prep = pipeline.named_steps["prep"]
            selector = pipeline.named_steps["select"]

            feature_names = prep.get_feature_names_out()
            total_features = len(feature_names)
            selected_features = selector.get_feature_names_out(feature_names).tolist()
            selected_min_votes = selector.min_votes

        # ----- Structured Report ----- #

        report = ClassificationReport(
            run_id=run_id,
            config=config,
            baseline_score=floor_scores["validation"].mean(),
            untuned_scores=untuned_scores,
            development_rows=len(X_dev),
            nested_scores=nested_scores,
            study=study,
            oof_true=oof_true,
            oof_proba=oof_proba,
            total_features=total_features,
            selected_features=selected_features,
            selected_min_votes=selected_min_votes,
        )

        row = report.to_scoreboard_row()
        runs = save_scores(config.get("output", {}).get("scoreboard", "results/runs.csv"), row)
        report.scoreboard = build_scoreboard(runs, row)
        path = save_run(config.get("output", {}).get("results_dir", "results"), run_id, config, str(report))

        print(f"CV {selection_metric}: {row['mean']:.3f} ± {row['std']:.3f}")
        print(f"Report saved: {path}")

        # ----- Final Classifier ----- #
        final_classifier = clone(pipeline).set_params(memory=None)
        
        # Train on all labeled observations, not just the dev set
        X_full, y_clf_full, _ = split_features_target(df_cleaned, preprocessing_config)
        valid = y_clf_full.isin([0, 1])
        final_classifier.fit(X_full.loc[valid], y_clf_full.loc[valid].astype(int))

        create_kaggle_submission(final_classifier, X_full, config)

        # This fitted classifier will then be used by the regression pipeline


if __name__ == "__main__":
    main()