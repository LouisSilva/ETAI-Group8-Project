import tempfile
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.pipeline import Pipeline

from .model import build_pipeline
from .evaluation import build_scorers, cross_validate_pipeline
from .tuning import tune_pipeline, nested_cross_validate
from .reports import ClassificationReport, RegressionReport
from .sklearn_extensions import CalendarHorizonSplit


@dataclass
class ExperimentResult:
    selected_pipeline: Pipeline
    report: ClassificationReport | RegressionReport


def run_classification_experiment(
    X_dev: pd.DataFrame,
    y_clf_dev: pd.Series,
    config: dict,
    run_id: str,
) -> ExperimentResult:
    """Evaluate and tune using development data only; return an unfitted pipeline."""
    preprocessing_config = config["preprocessing"]
    cv_config = config["cv"]
    selection_config = config["feature_selection"]
    tuning_config = config.get("tuning", {})

    selection_metric = cv_config["selection_metric"]
    scorers = build_scorers(cv_config["metrics"])
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

    with tempfile.TemporaryDirectory(prefix="pipeline_cache_") as cache_dir:
        pipeline = build_pipeline(
            preprocessing_config, 
            config["models"], 
            config["feature_selection"], 
            memory=cache_dir
        )
        pipeline.verbose = config.get("pipeline_verbose_enabled", False)

        # Baseline/floor CV
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

        # Untuned CV
        untuned_scores, _ = cross_validate_pipeline(
            pipeline,
            X_dev,
            y_clf_dev,
            outer_cv,
            scorers,
            selection_metric,
            n_jobs,
        )

        # Optional nested tuning
        study = None
        nested_scores = None
        oof_true = None
        oof_proba = None

        if tuning_config.get("enabled", False):
            selected_model = config["models"]["selected_model"]
            search_space = dict(config["tuning"]["search_spaces"].get(selected_model, {}))
            if not search_space:
                raise ValueError(f"Tuning is enabled, but the selected model '{selected_model!r}' has no search spaces defined in config.yaml.")

            if selection_config.get("enabled", False):
                search_space.update(selection_config.get("search_spaces", {}))

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

        # Fit on development data only to inspect selected feature names
        pipeline.fit(X_dev, y_clf_dev)
        selected_features = None
        total_features = None
        selected_min_votes = None

        if selection_config.get("enabled", False):
            feature_names = pipeline.named_steps["prep"].get_feature_names_out()
            selector = pipeline.named_steps["select"]
            selected_features = selector.get_feature_names_out(feature_names).tolist()
            total_features = len(feature_names)
            selected_min_votes = selector.min_votes

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

        # Remove the cache location and fitted state before returning
        selected_pipeline = clone(pipeline).set_params(memory=None)

    return ExperimentResult(selected_pipeline=selected_pipeline, report=report)

