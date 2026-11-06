from src.config import load_yaml_config
from src.data import load_data
from src.cleaning import clean_dataset
from src.preprocessing import split_features_target, train_validation_split
from src.sklearn_extensions import CalendarHorizonSplit
from src.model import build_pipeline
from src.evaluation import build_scorers, cross_validate_pipeline, cv_report

def main():
    config = load_yaml_config()

    # Load and clean the dataset
    df_raw = load_data(config["data"]["csv_paths"]["train"])
    df_cleaned = clean_dataset(df_raw, config["cleaning"]["train"])

    # Create a train and validation set for both the classification and regression task. clf = classification, reg = regression
    preprocessing_config = config["preprocessing"]
    X, y_clf, y_reg = split_features_target(df_cleaned, preprocessing_config)

    # TODO: rename "validation" in this part to dev, or just something else. Validation is the thing inside the cv folds, this bit is a locked test set.
    X_train, X_validation, y_clf_train, y_clf_validation, y_reg_train, y_reg_validation = train_validation_split(X, y_clf, y_reg, config["split"])

    preprocessing_config["numeric_features"] = X_train.select_dtypes(include="number").columns.tolist()
    preprocessing_config["categorical_features"] = X_train.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    pipeline = build_pipeline(preprocessing_config, config["models"], config["feature_selection"])

    cv_config = config["cv"]
    
    outer_cv = CalendarHorizonSplit(
        date_column="purchase_date",
        horizon_length_days=config["split"]["horizon_days"],
        validation_length_days=cv_config["validation_days"],
        n_splits=cv_config["outer_splits"]
    )

    # Sanity check that the CalendarHorizonSplit actually works properly
    for fold, (train_idx, val_idx) in enumerate(outer_cv.split(X_train, y_clf_train), start=1):
        train_dates = X_train.iloc[train_idx]["purchase_date"]
        val_dates = X_train.iloc[val_idx]["purchase_date"]
    
        print(f"\nFold {fold}")
        print(f"Training rows:   {len(train_idx)}")
        print(f"Validation rows: {len(val_idx)}")
        print(f"Latest training date: {train_dates.max()}")
        print(f"First validation date: {val_dates.min()}")
        print(f"Last validation date:  {val_dates.max()}")
    
        print(
            "Validation positive rate:",
            y_clf_train.iloc[val_idx].mean()
        )
    
        assert (
            train_dates.max() + pd.Timedelta(days=90)
            < val_dates.min()
        )

    # Not used yet
    inner_cv = CalendarHorizonSplit(
        date_column="purchase_date",
        horizon_length_days=config["split"]["horizon_days"],
        validation_length_days=cv_config["validation_days"],
        n_splits=cv_config["inner_splits"]
    )

    scorers = build_scorers(cv_config["metrics"])
    fold_scores, _ = cross_validate_pipeline(
        pipeline,
        X_train,
        y_clf_train,
        outer_cv,
        scorers,
        cv_config["selection_metric"],
        cv_config.get("n_jobs", 1),
    )

    report = cv_report(
        fold_scores,
        cv_config["metrics"],
        cv_config["selection_metric"],
        "Default model: chronological CV"
    )

    print(report)