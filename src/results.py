"""Saving each run: its scores to the runs table, and its full report to disk."""
import os

import pandas as pd


def save_scores(path: str, row: dict) -> pd.DataFrame:
    """Appends one run to the CSV at `path` (created if needed) and returns all runs."""
    runs = pd.read_csv(path, dtype={"run": str}) if os.path.exists(path) else pd.DataFrame()
    runs = pd.concat([runs, pd.DataFrame([row])], ignore_index=True)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    runs.to_csv(path, index=False)
    return runs


def save_run(results_dir: str, run: str, config: dict, report_text: str) -> str:
    """
    Writes the run's settings and full report to `results_dir/run_<run>.txt` (the folder is
    created if needed) and returns the path.
    """
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"run_{run}.txt")

    cv = config["cv"]
    tuning = config.get("tuning", {})
    tuning_line = (
        f"Optuna, {tuning['n_trials']} trials, "
        f"{cv['outer_splits']} outer × {cv['inner_splits']} inner folds "
        f"(nested chronological CV + final development-set tuning)  "
        f"random_state={tuning['random_state']}" if tuning.get("enabled", False) else "off"
    )
    
    selection = config.get("feature_selection", {})
    selection_line = (f"VoteSelector, min_votes={selection.get('min_votes', 2)} (before tuning)"
                      if selection.get("enabled", False) else "off")

    cv = config["cv"]
    split = config["split"]
    selected_model = config['models']['selected_model']
    header = (
        f"Run: {run}\n"
        f"Model: {selected_model}  "
        f"params={config['models'].get(selected_model, {})}\n"
        f"Preprocessing: encoder={config['preprocessing']['encoder']}  "
        f"scaler={config['preprocessing']['scaler']}\n"
        f"CV: chronological, {cv['outer_splits']} outer folds, "
        f"{cv['inner_splits']} inner folds\n"
        f"CV validation length: {cv['validation_length_days']} days  "
        f"horizon: {cv['horizon_length_days']} days\n"
        f"Metrics: {cv['metrics']}  "
        f"selection metric: {cv['selection_metric']}\n"
        f"Feature selection: {selection_line}\n"
        f"Tuning: {tuning_line}\n"
        f"Locked test set: size={split['test_size']:.0%}  "
        f"horizon: {split['horizon_length_days']} days  "
        f"(chronological, not evaluated)\n"
        + "=" * 60 + "\n\n"
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write(header + report_text)
        
    return path