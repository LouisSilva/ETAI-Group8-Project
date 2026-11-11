from typing import Any
from dataclasses import dataclass, field

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

    scoreboard: pd.DataFrame | None = None

    @property
    def final_scores(self) -> pd.DataFrame:
        return (
            self.nested_scores
            if self.nested_scores is not None
            else self.untuned_scores
        )

    def to_scoreboard_row(self) -> dict:
        """Structured results for runs.csv."""
        config = self.config
        model = config["models"]["selected_model"]
        metric = config["cv"]["selection_metric"]
        scores = self.final_scores["validation"]

        return {
            "run_id": self.run_id,
            "data": config["data"]["csv_paths"]["train"],
            "model": model,
            "tuned": self.study is not None,
            "encoder": config["preprocessing"]["encoder"],
            "scaler": config["preprocessing"]["scaler"],
            "feature_selection": (
                f"min_votes {self.selected_min_votes}"
                if self.selected_min_votes is not None
                else "off"
            ),
            "params": str(config["models"].get(model, {})),
            "tuned_params": (
                str(self.study.best_params)
                if self.study is not None else ""
            ),
            "metric": metric,
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
        config = self.config
        model = config["models"]["selected_model"]
        metrics = config["cv"]["metrics"]
        metric = config["cv"]["selection_metric"]

        sections = [
            f"Hyperparameters from config.yaml: "
            f"{config['models'].get(model, {})}",

            cv_report(
                self.untuned_scores,
                metrics,
                metric,
                "Cross-validation"
            ),

            f"No-model floor (dummy, same folds): "
            f"{metric} = {self.baseline_score:.3f}",
        ]

        if self.nested_scores is not None:
            sections.append(
                cv_report(
                    self.nested_scores,
                    metrics,
                    metric,
                    "Nested chronological CV"
                )
            )

        if self.study is not None:
            sections.append(tuning_report(self.study, self.nested_scores, metric))

        if self.oof_true is not None and self.oof_proba is not None:
            sections.append(oof_classification_report(self.oof_true, self.oof_proba))

        sections.append(
            f"Final model: {model} refit on all "
            f"{self.development_rows} development rows."
        )

        if self.selected_features is not None:
            sections.append(
                f"Features kept: {len(self.selected_features)} "
                f"of {self.total_features}:\n"
                + ", ".join(self.selected_features)
            )

        if self.scoreboard is not None:
            sections.append(
                "Scoreboard:\n"
                + self.scoreboard.to_string(
                    index=False,
                    float_format=lambda value: f"{value:.3f}"
                )
            )

        return "\n\n".join(sections)