import yaml
import os

def load_config_file(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Config file not found: {file_path}")

    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config