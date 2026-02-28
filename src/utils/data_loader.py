"""Shared data loading utilities for the Dynasty Model."""

import pandas as pd

from src.utils.config import load_config, resolve_path
from src.utils.constants import COLUMN_RENAME_MAP, COLUMNS_TO_DROP


def load_raw_dataset(path=None):
    """
    Loads the raw dataset with canonical column renames, junk column drops,
    and League Average row filtering applied.
    """
    if path is None:
        config = load_config()
        path = resolve_path(config["project"]["raw_data"])

    df = pd.read_csv(path)

    # Filter out League Average rows
    if "Player" in df.columns:
        df = df[df["Player"] != "League Average"].copy()

    # Drop junk columns
    cols_to_drop = [c for c in COLUMNS_TO_DROP if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    # Apply canonical renames
    rename_map = {k: v for k, v in COLUMN_RENAME_MAP.items() if k in df.columns}
    if rename_map:
        df = df.rename(columns=rename_map)

    return df


def load_processed_dataset(path=None):
    """Loads the processed dataset."""
    if path is None:
        config = load_config()
        path = resolve_path(config["project"]["processed_data"])

    return pd.read_csv(path)
