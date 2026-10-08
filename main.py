from src.config import load_yaml_config
from src.data import load_data
from src.preprocessing import clean_dataset

def main():
    config = load_yaml_config()

    df_train_raw = load_data(config["data"]["csv_paths"]["train"])
    df_train_cleaned = clean_dataset(df_train_raw, config["cleaning"])