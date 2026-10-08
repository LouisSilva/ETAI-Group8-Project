"""Data Loading."""
import pandas as pd
from pathlib import Path

def load_data(path: str | Path) -> pd.DataFrame:
    """Load the raw dataset from a CSV file."""
    return pd.read_csv(path)