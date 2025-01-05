from typing import Tuple, Optional
import pandas as pd
from pathlib import Path


def load_data(campaign: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Load train and test data for a given campaign.

    Args:
        campaign (str): The campaign name (e.g., "campaign1", "campaign2", "campaign3").

    Returns:
        tuple: (train_data, test_data) as DataFrames if files exist, otherwise (None, None).
    """
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    train_data_path = (
        BASE_DIR / "data" / "output_data" / campaign / "train_data.parquet"
    )
    test_data_path = BASE_DIR / "data" / "output_data" / campaign / "test_data.parquet"

    if train_data_path.exists() and test_data_path.exists():
        return pd.read_parquet(train_data_path), pd.read_parquet(test_data_path)

    return None, None
