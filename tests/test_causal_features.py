"""Causality tests for aggregation features.

The gold-standard property: a causal feature's value at row i must be
IDENTICAL when recomputed on the table truncated to rows [0..i]. Any feature
that peeks at the future fails this prefix-recomputation test.
"""
import numpy as np
import pandas as pd
import pytest

from src.features.build import causal_aggregates, row_local_features


def _frame():
    # one card seen at t=100, 200, 400; another card interleaved
    return pd.DataFrame({
        "TransactionDT": [100, 150, 200, 300, 400],
        "card1": [7, 9, 7, 9, 7],
        "addr1": [1.0, 2.0, 1.0, 2.0, 1.0],
        "TransactionAmt": [10.0, 99.0, 30.0, 42.0, 50.0],
    })


def test_past_count_counts_only_strict_past():
    out = causal_aggregates(_frame(), "card1", ["card1"])
    # card 7 appears at rows 0, 2, 4 -> past counts 0, 1, 2
    assert out["card1_past_count"].tolist() == [0.0, 0.0, 1.0, 1.0, 2.0]


def test_since_last_uses_previous_transaction():
    out = causal_aggregates(_frame(), "card1", ["card1"])
    v = out["card1_since_last"].to_numpy()
    assert np.isnan(v[0]) and np.isnan(v[1])          # first sightings
    assert v[2] == pytest.approx(100.0)               # 200 - 100
    assert v[4] == pytest.approx(200.0)               # 400 - 200


def test_past_mean_excludes_current_row():
    out = causal_aggregates(_frame(), "card1", ["card1"])
    m = out["card1_past_amt_mean"].to_numpy()
    assert np.isnan(m[0])
    assert m[2] == pytest.approx(10.0)                # mean of {10}
    assert m[4] == pytest.approx(20.0)                # mean of {10, 30}
    # ratio at last row: 50 / 20
    assert out["card1_amt_ratio"].to_numpy()[4] == pytest.approx(2.5)


def test_prefix_recomputation_property():
    """Feature at row i is unchanged when the future is deleted."""
    df = _frame()
    full = causal_aggregates(df, "card1", ["card1"])
    for i in range(len(df)):
        prefix = causal_aggregates(df.iloc[:i + 1].copy(), "card1", ["card1"])
        for col in full.columns:
            a, b = full[col].iloc[i], prefix[col].iloc[i]
            assert (pd.isna(a) and pd.isna(b)) or a == pytest.approx(b), \
                f"{col} at row {i} changed when future rows were removed"


def test_prefix_property_combo_key():
    df = _frame()
    full = causal_aggregates(df, "card1_addr1", ["card1", "addr1"])
    for i in range(len(df)):
        prefix = causal_aggregates(df.iloc[:i + 1].copy(), "card1_addr1",
                                   ["card1", "addr1"])
        for col in full.columns:
            a, b = full[col].iloc[i], prefix[col].iloc[i]
            assert (pd.isna(a) and pd.isna(b)) or a == pytest.approx(b)


def test_row_local_features_are_row_local():
    df = _frame()
    full = row_local_features(df)
    prefix = row_local_features(df.iloc[:2].copy())
    pd.testing.assert_frame_equal(full.iloc[:2], prefix)


@pytest.mark.skipif(
    not ((__import__("src.config", fromlist=["config"]).DATA_PROCESSED
          / "features.parquet").exists()),
    reason="features not built")
def test_real_features_prefix_sample():
    """Spot-check the prefix property on the real table (random cut points)."""
    from src import config
    cols = ["TransactionDT", "card1", "addr1", "TransactionAmt"]
    df = pd.read_parquet(config.DATA_PROCESSED / "train.parquet", columns=cols)
    feats = pd.read_parquet(config.DATA_PROCESSED / "features.parquet",
                            columns=["card1_past_count", "card1_past_amt_mean"])
    rng = np.random.RandomState(0)
    for cut in rng.randint(1000, len(df), size=3):
        prefix = causal_aggregates(df.iloc[:cut].copy(), "card1", ["card1"])
        got = feats.iloc[cut - 1]
        want = prefix.iloc[cut - 1]
        for col in ("card1_past_count", "card1_past_amt_mean"):
            a, b = got[col], want[col]
            assert (pd.isna(a) and pd.isna(b)) or a == pytest.approx(b)
