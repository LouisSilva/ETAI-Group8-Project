"""Config Loading."""
import yaml
from pathlib import Path

def load_yaml_config(path: str | Path = "config.yaml") -> dict:
    """Load the YAML config from a .yaml file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)