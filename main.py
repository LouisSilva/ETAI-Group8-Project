import shutil
import tempfile

import pandas as pd

from src.config import load_yaml_config
from src.data import load_data
from src.cleaning import clean_dataset
from src.preprocessing import split_features_target, development_test_split
from src.sklearn_extensions import CalendarHorizonSplit
from src.model import build_pipeline
from src.evaluation import build_scorers, cross_validate_pipeline, cv_report
from src.tuning import tune_pipeline, nested_cross_validate, tuning_report

def main():
    config = load_yaml_config()

    # Load and clean the dataset
    df_raw = load_data(config["data"]["csv_paths"]["train"])
    df_cleaned = clean_dataset(df_raw, config["cleaning"]["train"])

    # Create a train and validation set for both the classification and regression task. clf = classification, reg = regression
    preprocessing_config = config["preprocessing"]
    X, y_clf, y_reg = split_features_target(df_cleaned, preprocessing_config)

    # Every decision uses the development set only; the test set stays locked
    X_dev, X_test, y_clf_dev, y_clf_test, y_reg_dev, y_reg_test = development_test_split(X, y_clf, y_reg, config["split"])

    preprocessing_config["numeric_features"] = X_dev.select_dtypes(include="number").columns.tolist()
    preprocessing_config["categorical_features"] = X_dev.select_dtypes(include=["object", "category", "string"]).columns.tolist()

    with tempfile.TemporaryDirectory(prefix="pipeline_cache_") as cache_dir:
    
        pipeline = build_pipeline(preprocessing_config, config["models"], config["feature_selection"], memory=cache_dir)
        pipeline.verbose = config.get("pipeline_verbose_enabled", False)
    
        # ----- Cross-Validate Selected Model -----
    
        cv_config = config["cv"]
        selection_metric = cv_config["selection_metric"]
        metrics = cv_config["metrics"]
        
        outer_cv = CalendarHorizonSplit(
            date_column="purchase_date",
            horizon_length_days=cv_config["horizon_length_days"],
            validation_length_days=cv_config["validation_length_days"],
            n_splits=cv_config["outer_splits"]
        )
    
        # Not used yet
        inner_cv = CalendarHorizonSplit(
            date_column="purchase_date",
            horizon_length_days=cv_config["horizon_length_days"],
            validation_length_days=cv_config["validation_length_days"],
            n_splits=cv_config["inner_splits"]
        )
    
        scorers = build_scorers(metrics)
        fold_scores, _ = cross_validate_pipeline(
            pipeline,
            X_dev,
            y_clf_dev,
            outer_cv,
            scorers,
            selection_metric,
            cv_config.get("n_jobs", 1),
        )
    
        report = cv_report(
            fold_scores,
            metrics,
            selection_metric,
            "Default model: chronological CV"
        )
    
        # ----- Hyperparameter Tuning -----
    
        tuning_config = config["tuning"]
        if tuning_config["enabled"]:
            selected_model = config["models"]["selected_model"]
            search_space = dict(tuning_config["search_spaces"][selected_model])
    
            if config["feature_selection"]["enabled"]:
                search_space.update(config["feature_selection"].get("search_space", {}))
    
            fold_scores, _ = nested_cross_validate(
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
                n_jobs=cv_config.get("n_jobs", 1)
            )
    
            report += "\n\n" + cv_report(
                fold_scores,
                metrics,
                selection_metric,
                "Nested chronological CV"
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
                n_jobs=cv_config.get("n_jobs", 1)
            )
    
            report += "\n\n" + tuning_report(
                study,
                fold_scores,
                selection_metric
            )
        
        print(report)

        # pipeline.fit

        pipeline.memory = None
        shutil.rmtree(cache_dir, ignore_errors=True)


if __name__ == "__main__":
    main()