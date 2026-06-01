import pandas as pd
from pathlib import Path


def load_csv(path: str) -> pd.DataFrame:
    path = Path(path)
    return pd.read_csv(path)
