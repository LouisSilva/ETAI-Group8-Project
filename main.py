import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import joblib

from sklearn.base import clone

from src.config import load_yaml_config
from src.data import load_data
from src.cleaning import clean_dataset
from src.preprocessing import split_features_target, development_test_split
from src.sklearn_extensions import CalendarHorizonSplit
from src.model import build_pipeline
from src.evaluation import build_scorers, cross_validate_pipeline, cv_report, oof_classification_report
from src.tuning import tune_pipeline, nested_cross_validate, tuning_report
from src.results import save_run, save_scores
from src.kaggle_submission import create_kaggle_submission


def main():
    config = load_yaml_config()
    run = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load and clean the dataset
    df_raw = load_data(config["data"]["csv_paths"]["train"])
    df_cleaned = clean_dataset(df_raw, config["cleaning"]["train"])
    # df_cleaned = engineer_features(df_cleaned)

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
        
        fold_scores, oof_proba = cross_validate_pipeline(
            pipeline,
            X_dev,
            y_clf_dev,
            outer_cv,
            scorers,
            selection_metric,
            n_jobs,
        )

        floor, untuned = floor_scores["validation"].mean(), fold_scores["validation"].mean()

        print("\nDefault hyperparameters (config.yaml)")
        print(f"  {selection_metric}: {untuned:.3f} ± {fold_scores['validation'].std(ddof=1):.3f}   "
              f"train-validation gap: {fold_scores['gap'].mean():+.3f}   no-model floor: {floor:.3f}")
        report = (f"Hyperparameters from config.yaml: {config["models"].get(selected_model, {})}\n\n"
                  + cv_report(fold_scores, metrics, selection_metric, "Cross-validation")
                  + f"\n\nNo-model floor (dummy, same folds): {selection_metric} = {floor:.3f}")
        final_scores = fold_scores
    
        # ----- Hyperparameter Tuning ----- #
    
        tuning_config = config.get("tuning", {})
        tuning_enabled = tuning_config.get("enabled", False)
        selection_enabled = config["feature_selection"].get("enabled", False)
        if tuning_enabled:
            search_space = dict(tuning_config["search_spaces"][selected_model])
    
            if selection_enabled:
                search_space.update(config["feature_selection"].get("search_spaces", {}))

            print(f"\nTuning {tuning_config["n_trials"]} trials, nested CV", flush=True)
            final_scores, oof_proba = nested_cross_validate(
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
    
            report += "\n\n" + cv_report(
                final_scores,
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
                n_jobs=n_jobs
            )

            print("  Chosen hyperparameters:")
            for name, value in study.best_params.items():
                print(f"    {name.replace('model__', ''):<20s} {value:.4g}" if isinstance(value, float)
                      else f"    {name.replace('model__', ''):<20s} {value}")
    
            report += "\n\n" + tuning_report(
                study,
                final_scores,
                selection_metric
            )

            print(f"\nAll metrics ({'tuned model' if tuning_enabled else 'default hyperparameters'}, mean ± std)")
            for name in metrics:
                print(f"  {name:<18s} {final_scores[name].mean():.3f} ± {final_scores[name].std(ddof=1):.3f}")

            valid = np.isfinite(oof_proba)
            report += "\n\n" + oof_classification_report(y_clf_dev.iloc[valid], oof_proba[valid])

        # ----- Fit Final Pipeline ----- #
        pipeline.fit(X_dev, y_clf_dev)
        refit = f"Final model: {selected_model} refit on all {len(X_dev)} development rows."
        if selection_enabled:
            all_features = pipeline.named_steps["prep"].get_feature_names_out()
            kept = pipeline.named_steps["select"].get_feature_names_out(all_features)
            refit += f"\nFeatures kept: {len(kept)} of {len(all_features)}: {', '.join(kept)}"
        report += "\n" + refit + "\n"

        row = {
            "run": run, 
            "data": config["data"]["csv_paths"]["train"], 
            "model": selected_model, 
            "tuned": tuning_enabled,
            "encoder": config["preprocessing"]["encoder"], 
            "scaler": config["preprocessing"]["scaler"],
            "feature_selection": (f"min_votes {pipeline.named_steps['select'].min_votes}" if selection_enabled else "off"),
            "params": str(config["models"].get(selected_model, {})), 
            "tuned_params": str(study.best_params) if tuning_enabled else "",
            "metric": selection_metric, 
            "mean": final_scores["validation"].mean(),
            "std": final_scores["validation"].std(ddof=1), 
            "untuned": untuned, 
            "floor": floor,
           **{f"fold_{i}": score for i, score in enumerate(final_scores["validation"], start=1)}
        }
        runs = save_scores(config.get("output", {}).get("scoreboard", "results/runs.csv"), row)
        same = runs[(runs["data"] == row["data"]) & (runs["metric"] == selection_metric)]
        board = (same.sort_values("mean", ascending=False)
                 [["run", "model", "tuned", "feature_selection", "mean", "std", "untuned", "floor"]]
                 .rename(columns={"mean": selection_metric})
                 .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print(f"\nScoreboard (runs on the same data and metric; best first)\n{board}")
        report += f"\nScoreboard:\n{board}\n"

        shutil.rmtree(cache_dir, ignore_errors=True)
        path = save_run(config.get("output", {}).get("results_dir", "results"), run, config, report)
        print(f"\n{refit}\nFull report (per-fold tables, tuning, confusion matrix, fairness): {path}")

        # ----- Kaggle Submission ----- #
        final_pipeline = clone(pipeline).set_params(memory=None)

        # Train on all labeled observations
        X_full, y_clf_full, _ = split_features_target(df_cleaned, preprocessing_config)
        valid = y_clf_full.isin([0, 1])
        final_pipeline.fit(X_full.loc[valid], y_clf_full.loc[valid].astype(int))

        create_kaggle_submission(final_pipeline, X_full, config)


if __name__ == "__main__":
    main()