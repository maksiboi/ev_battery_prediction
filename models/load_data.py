import pandas as pd
from pathlib import Path

def load_data_campaign1():
    BASE_DIR = Path(__file__).resolve().parent.parent
    train_data_path = BASE_DIR / "data" / "output_data" / "campaign1" / "train_data.parquet"
    test_data_path = BASE_DIR / "data" / "output_data" / "campaign1" / "test_data.parquet"

    if train_data_path.exists() and test_data_path.exists():
        return pd.read_parquet(train_data_path), pd.read_parquet(test_data_path)
    return None, None

def load_data_campaign2():
    BASE_DIR = Path(__file__).resolve().parent.parent
    train_data_path = BASE_DIR / "data" / "output_data" / "campaign2" / "train_data.parquet"
    test_data_path = BASE_DIR / "data" / "output_data" / "campaign2" / "test_data.parquet"

    if train_data_path.exists() and test_data_path.exists():
        return pd.read_parquet(train_data_path), pd.read_parquet(test_data_path)
    return None, None

def load_data_campaign3():
    BASE_DIR = Path(__file__).resolve().parent.parent
    train_data_path = BASE_DIR / "data" / "output_data" / "campaign3" / "train_data.parquet"
    test_data_path = BASE_DIR / "data" / "output_data" / "campaign3" / "test_data.parquet"

    if train_data_path.exists() and test_data_path.exists():
        return pd.read_parquet(train_data_path), pd.read_parquet(test_data_path)
    return None, None