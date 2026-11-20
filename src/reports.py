from typing import Any
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent

import numpy as np
import pandas as pd

from .evaluation import cv_report, oof_classification_report
from .tuning import tuning_report


@dataclass
class ClassificationReport:
    run_id: str
    config: dict
    baseline_score: float
    untuned_scores: pd.DataFrame
    development_rows: int

    nested_scores: pd.DataFrame | None = None
    study: Any | None = None
    oof_true: pd.Series | None = None
    oof_proba: np.ndarray | None = None
    total_features: int | None = None
    selected_features: list[str] | None = None
    selected_min_votes: int | None = None

    # saved_model_path: Path | None = None

    # ----- Config property getters for ease of use (might remove later) ----- #
    @property
    def prep_cfg(self) -> dict:
        return self.config["preprocessing"]

    @property
    def cv_cfg(self) -> dict:
        return self.config["cv"]

    @property
    def models_cfg(self) -> dict:
        return self.config["models"]

    @property
    def f_selection_cfg(self) -> dict:
        return self.config["feature_selection"]

    @property
    def tuning_cfg(self) -> dict:
        return self.config["tuning"]

    @property
    def model_name(self) -> str:
        return self.models_cfg["selected_model"]

    @property
    def final_scores(self) -> pd.DataFrame:
        return (
            self.nested_scores
            if self.nested_scores is not None
            else self.untuned_scores
        )

    def to_scoreboard_row(self) -> dict:
        """Structured results for runs.csv."""
        scores = self.final_scores["validation"]

        return {
            "run_id": self.run_id,
            "task": "classification",
            "data": self.config["data"]["csv_paths"]["train"],
            "model": self.model_name,
            "tuned": self.study is not None,
            "encoder": self.prep_cfg["encoder"],
            "scaler": self.prep_cfg["scaler"],
            "numerical_imputer": self.prep_cfg["numeric_imputation"].get("selected_imputer", "simple_imputer"),
            "categorical_imputer": self.prep_cfg["categorical_imputation"].get("selected_imputer", "simple_imputer"),
            "feature_selection": (
                f"min_votes {self.selected_min_votes}"
                if self.selected_min_votes is not None
                else "off"
            ),
            "params": str(self.models_cfg.get(self.model_name, {})),
            "tuned_params": (
                str(self.study.best_params)
                if self.study is not None else ""
            ),
            "metric": self.cv_cfg["selection_metric"],
            "mean": scores.mean(),
            "std": scores.std(ddof=1),
            "untuned": self.untuned_scores["validation"].mean(),
            "floor": self.baseline_score,
            **{
                f"fold_{i}": score
                for i, score in enumerate(scores, start=1)
            },
        }

    def __str__(self) -> str:
        feature_selection_text = (
            f"VoteSelector, min_votes={self.f_selection_cfg.get('min_votes', 2)}"
            if self.f_selection_cfg.get("enabled", False) 
            else "DISABLED"
        )

        hyperparameter_tuning_text = (
            f"Optuna, {self.tuning_cfg['n_trials']} trials"
            if self.tuning_cfg.get("enabled", False) 
            else "DISABLED"
        )

        untuned_cv_report = cv_report(
            self.untuned_scores, 
            self.cv_cfg["metrics"], 
            self.cv_cfg["selection_metric"],
        )

        nested_cv_report = (
            cv_report(
                self.nested_scores,
                self.cv_cfg["metrics"], 
                self.cv_cfg["selection_metric"],
            )
            if self.nested_scores is not None
            else "DISABLED"
        )

        hyperparameter_tuning_report = (
            tuning_report(
                self.study,
                self.nested_scores,
                self.cv_cfg["selection_metric"]
            )
            if self.study is not None
            else "DISABLED"
        )

        oof_report = (
            oof_classification_report(self.oof_true, self.oof_proba)
            if self.oof_true is not None and self.oof_proba is not None
            else "DISABLED"
        )

        selected_features_text = (
            f"{len(self.selected_features)} of {self.total_features}:\n" + ", ".join(self.selected_features)
            if self.selected_features is not None
            else "All"
        )

        return dedent(f"""
Run ID: {self.run_id}
Task: Classification

Model: {self.model_name}
Default hyperparameters: {self.models_cfg.get(self.model_name, {})}

Preprocessing:
    - Encoder: {self.prep_cfg["encoder"]}
    - Scaler: {self.prep_cfg["scaler"]}
    - Numeric imputer: {self.prep_cfg["numeric_imputation"]["selected_imputer"]}
    - Categorical imputer: {self.prep_cfg["categorical_imputation"]["selected_imputer"]}

Cross-Validation:
    - Outer folds: {self.cv_cfg['outer_splits']}
    - Inner folds: {self.cv_cfg['inner_splits']}
    - Validation window: {self.cv_cfg['validation_length_days']} days
    - Prediction horizon: {self.cv_cfg['horizon_length_days']} days
    - Metrics: {self.cv_cfg['metrics']}
    - Selection metric: {self.cv_cfg['selection_metric']}

Feature selection: {feature_selection_text}
Hyperparameter tuning: {hyperparameter_tuning_text}

{"=" * 60}

Untuned Chronological CV Report {untuned_cv_report}

Dummy model floor achieved {self.cv_cfg['selection_metric']} = {self.baseline_score:.3f}

{"=" * 60}

Nested Chronological CV Report {nested_cv_report}

{"=" * 60}

Hyperparameter Tuning Report: {hyperparameter_tuning_report}

{"=" * 60}

Features kept: {selected_features_text}

{"=" * 60}

Out-of-Fold Classification Report: {oof_report}

{"=" * 60}
""").strip()


@dataclass
class RegressionReport:
    run_id: str
    config: dict


class Scoreboard:
    """Comparison of classification or regression runs."""
    runs: pd.DataFrame
    task: str

    def __init__(self, task: str, path: str | Path = "./results/runs.csv") -> None:
        self.task = task
        self.path = Path(path)

        self.load_runs_from_path()

    def load_runs_from_path(self) -> None:
        """
        Tries to load previous runs from the CSV located at `self.path` into a DataFrame.
        It will create an empty DataFrame if the CSV doesn't exist."""
        self.runs = (
            pd.read_csv(self.path, dtype={"run_id": str}) 
            if self.path.exists()
            else pd.DataFrame()
        )

    def save_score(self, new_scoreboard_row: dict) -> None:
        """Appends one run (`new_scoreboard_row`) to the CSV at `self.path` (created if needed)."""
        self.runs = pd.concat([self.runs, pd.DataFrame([new_scoreboard_row])], ignore_index=True)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.runs.to_csv(self.path, index=False)

    @property
    def table(self) -> pd.DataFrame:
        """Returns the sorted `runs` DataFrame of runs with task = `self.task`."""
        matching = self.runs.loc[self.runs["task"].eq(self.task)]
        columns = [
            "run_id",
            "model",
            "tuned",
            "encoder",
            "scaler",
            "numerical_imputer",
            "feature_selection",
            "mean",
            "std",
        ]

        return matching.sort_values("mean", ascending=False)[columns]
    
    def __str__(self) -> str:
        if self.table.empty:
            return "No comparable runs."

        return self.table.to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}"
        )


def save_report(
    path: str | Path, 
    report: ClassificationReport | RegressionReport,
    scoreboard: Scoreboard | None = None
) -> Path:
    """Write the report's text representation and optionally, the scoreboard. Returns the path where the report was written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    sections = [str(report)]
    if scoreboard is not None:
        sections.append(f"Scoreboard:\n {str(scoreboard)}")

    path.write_text("\n\n".join(sections), encoding="utf-8")

    return path
