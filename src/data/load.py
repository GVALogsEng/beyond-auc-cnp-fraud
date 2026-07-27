"""Loading helpers for the processed parquet artifacts."""
from __future__ import annotations

import pandas as pd

from src import config

META_COLS = ["TransactionID", "isFraud", "TransactionDT", "TransactionAmt",
             "day", "week", "month"]


def load_train(columns: list[str] | None = None) -> pd.DataFrame:
    """Load the joined training table (sorted by TransactionDT).

    `columns` may omit meta columns; they are always included.
    """
    path = config.DATA_PROCESSED / "train.parquet"
    if columns is not None:
        columns = list(dict.fromkeys(META_COLS + columns))
    return pd.read_parquet(path, columns=columns)


def load_features() -> pd.DataFrame:
    """Load the engineered feature table produced by src.features.build."""
    return pd.read_parquet(config.DATA_PROCESSED / "features.parquet")


def load_feature_columns(columns: list[str]) -> pd.DataFrame:
    columns = list(dict.fromkeys(META_COLS + columns))
    return pd.read_parquet(config.DATA_PROCESSED / "features.parquet",
                           columns=columns)


def split_masks(df: pd.DataFrame):
    """Boolean masks for the fixed temporal protocol."""
    train = df["month"].isin(config.TRAIN_MONTHS).to_numpy()
    val = (df["month"] == config.VAL_MONTH).to_numpy()
    test = (df["month"] == config.TEST_MONTH).to_numpy()
    return train, val, test
