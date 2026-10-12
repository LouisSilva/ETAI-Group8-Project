from src.config import load_yaml_config
from src.data import load_data
from src.cleaning import clean_dataset
from src.preprocessing import split_features_target, train_validation_split

def main():
    config = load_yaml_config()

    # Load and clean the dataset
    df_raw = load_data(config["data"]["csv_paths"]["train"])
    df_cleaned = clean_dataset(df_raw, config["cleaning"]["train"])

    # Create a train and validation set for both the classification and regression task. clf = classification, reg = regression
    X, y_clf, y_reg = split_features_target(df_cleaned, config["data"])
    X_train, X_validation, y_clf_train, y_clf_validation, y_reg_train, y_reg_validation = train_validation_split(X, y_clf, y_reg, config["split"])