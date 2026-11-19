from datetime import datetime
from pathlib import Path
import argparse
import joblib

from src.config import load_yaml_config
from src.data import load_data
from src.cleaning import clean_dataset
from src.preprocessing import split_features_target, development_test_split
from src.experiments import run_classification_experiment
from src.reports import Scoreboard, save_report
from src.kaggle_submission import create_kaggle_submission


def train_classification(is_save_model_enabled: bool, model_path: Path | None) -> None:
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
    new_scoreboard_row = report.to_scoreboard_row()

    output_cfg = config.get("output", {})
    results_dir = Path(output_cfg.get("results_dir", "results"))

    scoreboard = Scoreboard(task="classification", path=results_dir / "runs.csv")
    scoreboard.save_score(new_scoreboard_row)

    report_path = save_report(
        path=results_dir / f"run_{run_id}.txt",
        report=report,
        scoreboard=scoreboard
    )

    print(f"CV: {new_scoreboard_row['mean']:.3f} ± {new_scoreboard_row['std']:.3f}")
    print(f"Report saved to {report_path}")

    if not is_save_model_enabled: return

    # ----- Fit final classifier on all labelled train.csv observations ----- #
    model_path = Path(model_path or config["output"].get("model_path", "./models/model.joblib"))
    model_path.parent.mkdir(parents=True, exist_ok=True)

    final_classifier = experiment.selected_pipeline
    labelled = y_clf.isin([0, 1])
    final_classifier.fit(X.loc[labelled], y_clf.loc[labelled].astype(int))

    joblib.dump(final_classifier, model_path)
    print(f"Model saved to {model_path}")


def train_regression(is_save_model_enabled: bool, model_path: Path | None) -> None:
    pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ETAI Group 8 - Customer Repurchase Prediction"
    )

    commands = parser.add_subparsers(dest="command", required=True)

    # ----- Model Training Commands ----- #
    train_parser = commands.add_parser("train", aliases=["t"], help="Train a classification or regression model and tune its pipeline")
    train_tasks = train_parser.add_subparsers(dest="task", required="True")
    train_options = argparse.ArgumentParser(add_help=False)

    train_options.add_argument(
        "--no-save", "--no_save", "-n",
        dest="save",
        action="store_false",
        help="Do not save the fitted model to disk"
    )

    train_options.add_argument(
        "--out", "-o", "--model_path",
        dest="model_path",
        type=Path,
        default=None,
        help="Path to save the model; this option overrides the directory set in the YAML config"
    )

    train_tasks.add_parser(
        "classification",
        aliases=["classifier", "clf", "c"],
        parents=[train_options],
        help="Train the classification pipeline"
    ).set_defaults(func=lambda args: train_classification(
        is_save_model_enabled=args.save, 
        model_path=args.model_path))

    train_tasks.add_parser(
        "regression",
        aliases=["regressor", "reg", "r"],
        parents=[train_options],
        help="Train the regression pipeline"
    ).set_defaults(func=lambda args: train_regression(
        is_save_model_enabled=args.save, 
        model_path=args.model_path))

    # ----- predict.py Commands ----- #
    # Here we will make the commands interface with predict.py, (copy the base pipeline). It will be made so you can tell it to only output the classification or regression result, or both (which will be used for the kaggle submission, cuz we don't want the regression result in that). The regression model would only be ran on observations where the classification model predicted `repeat_thingybob_90d` == 1

    # ----- Client Interface Commands ----- #
    # Read the project pdf guide
    # The regression model would only be ran on observations where the classification model predicted `repeat_thingybob_90d` == 1

    args = parser.parse_args()
    args.func(args)


# ----- Create Kaggle submission ----- #
# create_kaggle_submission(final_classifier, X, config)


if __name__ == "__main__":
    main()



