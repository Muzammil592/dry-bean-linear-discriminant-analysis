"""Data loading helpers for the Dry Bean dataset."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

TARGET = "Class"


def load_dry_bean(path: str | Path = "data/raw/Dry_Bean_Dataset.xlsx"):
    """Load the UCI Dry Bean dataset (.xlsx or .csv) and return (X, y).

    X : DataFrame with the 16 geometric features
    y : Series with the 7 bean varieties
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. See data/README.md for download instructions."
        )
    df = pd.read_excel(path) if path.suffix in {".xlsx", ".xls"} else pd.read_csv(path)
    return df.drop(columns=[TARGET]), df[TARGET]
