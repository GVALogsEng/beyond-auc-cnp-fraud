"""Split-integrity tests: the temporal protocol admits no overlap."""
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold

from src import config
from src.data.load import split_masks
from src.data.prepare import add_time_index


def _synthetic_frame(n=10_000, span_days=182, seed=0):
    rng = np.random.RandomState(seed)
    dt = np.sort(rng.randint(config.DAY_SECONDS,
                             span_days * config.DAY_SECONDS, n))
    df = pd.DataFrame({"TransactionDT": dt})
    return add_time_index(df)


def test_month_indices_bounded():
    df = _synthetic_frame()
    assert df["month"].min() >= 0
    assert df["month"].max() == config.N_MONTHS - 1  # stub merged into month 5


def test_temporal_split_disjoint_and_ordered():
    df = _synthetic_frame()
    train, val, test = split_masks(df)
    assert not (train & val).any()
    assert not (train & test).any()
    assert not (val & test).any()
    # every train timestamp strictly precedes every val timestamp, etc.
    assert df.loc[train, "TransactionDT"].max() < df.loc[val, "TransactionDT"].min()
    assert df.loc[val, "TransactionDT"].max() < df.loc[test, "TransactionDT"].min()


def test_groupkfold_month_disjoint():
    df = _synthetic_frame()
    tr_mask = df["month"].isin(config.TRAIN_MONTHS).to_numpy()
    months = df.loc[tr_mask, "month"].to_numpy()
    X = np.zeros((tr_mask.sum(), 1))
    gkf = GroupKFold(n_splits=len(np.unique(months)))
    for tr_i, va_i in gkf.split(X, groups=months):
        # validation fold months never appear in the training fold
        assert set(months[tr_i]).isdisjoint(set(months[va_i]))


def test_day_week_month_consistency():
    df = _synthetic_frame()
    assert (df["day"] == df["TransactionDT"] // config.DAY_SECONDS).all()
    assert (df["week"] == df["day"] // config.WEEK_DAYS).all()
    expected_month = np.minimum(df["day"] // config.DAYS_PER_MONTH,
                                config.N_MONTHS - 1)
    assert (df["month"] == expected_month).all()


@pytest.mark.skipif(not (config.DATA_PROCESSED / "train.parquet").exists(),
                    reason="processed data not present")
def test_real_data_split_integrity():
    df = pd.read_parquet(config.DATA_PROCESSED / "train.parquet",
                         columns=["TransactionDT", "day", "month"])
    train, val, test = split_masks(df)
    assert train.sum() > 0 and val.sum() > 0 and test.sum() > 0
    assert df.loc[train, "TransactionDT"].max() < df.loc[val, "TransactionDT"].min()
    assert df.loc[val, "TransactionDT"].max() < df.loc[test, "TransactionDT"].min()
    assert not (train & val).any() and not (val & test).any()
