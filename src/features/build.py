"""Feature engineering with strict causality.

Every aggregation is an *expanding, past-only* statistic: the value attached
to a transaction uses only transactions that occur strictly earlier in the
table (which is sorted by TransactionDT; ties broken by stable row order).
This mirrors how a production feature store accumulates state and is enforced
by tests/test_causal_features.py via a prefix-recomputation property test.

Deliberate exclusions (DECISIONS.md D-004):
  * no label/target-based aggregations -- chargeback labels arrive weeks late
    in production, so past-label features would be silently optimistic;
  * no full UID reconstruction a la the 1st-place Kaggle solution -- cited in
    the leakage audit as precedent, but out of scope for an evaluation paper.

Feature blocks (Stage 6 ablation unit):
  A_core        amount, product, card, address, distance, email, time-of-day,
                and causal velocity/aggregate features
  B_counts      C1-C14
  C_timedeltas  D1-D15
  D_matches     M1-M9
  E_vesta       V1-V339
  F_identity    id_01-id_38, DeviceType, DeviceInfo, has_identity
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.data.load import META_COLS, load_train
from src.utils import get_logger, save_json, timer

log = get_logger("features.build")

AGG_KEYS = {"card1": ["card1"], "card1_addr1": ["card1", "addr1"]}


def causal_aggregates(df: pd.DataFrame, key_name: str, key_cols: list[str]) -> pd.DataFrame:
    """Expanding past-only aggregates for one entity key.

    Returns a frame with:
      <key>_past_count      number of strictly earlier transactions by entity
      <key>_since_last      seconds since the entity's previous transaction
      <key>_past_amt_mean   mean amount of the entity's earlier transactions
      <key>_amt_ratio       current amount / past mean amount
    """
    if len(key_cols) == 1:
        key = df[key_cols[0]].astype("str")
    else:
        key = df[key_cols[0]].astype("str")
        for c in key_cols[1:]:
            key = key + "_" + df[c].astype("str")

    g_dt = df.groupby(key, sort=False)["TransactionDT"]
    g_amt = df.groupby(key, sort=False)["TransactionAmt"]

    past_count = g_dt.cumcount().to_numpy(dtype="float32")
    since_last = (df["TransactionDT"] - g_dt.shift(1)).to_numpy(dtype="float32")
    amt = df["TransactionAmt"].to_numpy(dtype="float64")
    past_sum = g_amt.cumsum().to_numpy(dtype="float64") - amt
    with np.errstate(invalid="ignore", divide="ignore"):
        past_mean = np.where(past_count > 0, past_sum / past_count, np.nan)
        amt_ratio = np.where(past_count > 0, amt / past_mean, np.nan)

    out = pd.DataFrame({
        f"{key_name}_past_count": past_count,
        f"{key_name}_since_last": since_last,
        f"{key_name}_past_amt_mean": past_mean.astype("float32"),
        f"{key_name}_amt_ratio": amt_ratio.astype("float32"),
    }, index=df.index)
    return out


def row_local_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features computable from the row alone (trivially causal)."""
    out = pd.DataFrame(index=df.index)
    amt = df["TransactionAmt"].astype("float64")
    out["amt_log"] = np.log1p(amt).astype("float32")
    out["amt_cents"] = (amt - np.floor(amt)).round(2).astype("float32")
    out["hour"] = ((df["TransactionDT"] // 3600) % 24).astype("int8")
    out["dow"] = ((df["TransactionDT"] // config.DAY_SECONDS) % 7).astype("int8")
    return out


def block_columns(df: pd.DataFrame, engineered: list[str]) -> dict[str, list[str]]:
    cols = set(df.columns)
    blocks = {
        "A_core": (["TransactionAmt", "ProductCD", "card1", "card2", "card3",
                    "card4", "card5", "card6", "addr1", "addr2", "dist1",
                    "dist2", "P_emaildomain", "R_emaildomain"] + engineered),
        "B_counts": [f"C{i}" for i in range(1, 15)],
        "C_timedeltas": [f"D{i}" for i in range(1, 16)],
        "D_matches": [f"M{i}" for i in range(1, 10)],
        "E_vesta": [f"V{i}" for i in range(1, 340)],
        "F_identity": ([f"id_{i:02d}" for i in range(1, 39)]
                       + ["DeviceType", "DeviceInfo", "has_identity"]),
    }
    return {name: [c for c in cs if c in cols or c in engineered]
            for name, cs in blocks.items()}


def main() -> None:
    with timer("load train.parquet", log):
        df = load_train()
    assert df["TransactionDT"].is_monotonic_increasing, "table must be time-sorted"

    parts = [row_local_features(df)]
    engineered = list(parts[0].columns)
    for key_name, key_cols in AGG_KEYS.items():
        with timer(f"causal aggregates: {key_name}", log):
            agg = causal_aggregates(df, key_name, key_cols)
            parts.append(agg)
            engineered += list(agg.columns)

    feats = pd.concat([df] + parts, axis=1)
    blocks = block_columns(feats, engineered)

    manifest = {"blocks": blocks,
                "n_features_total": sum(len(v) for v in blocks.values()),
                "engineered_causal": engineered,
                "meta_columns": META_COLS}
    save_json(manifest, "feature_blocks")

    keep = META_COLS + [c for cols in blocks.values() for c in cols]
    keep = list(dict.fromkeys(keep))
    with timer("write features.parquet", log):
        feats[keep].to_parquet(config.DATA_PROCESSED / "features.parquet", index=False)
    log.info("features.parquet: %d rows x %d cols", len(feats), len(keep))

    for name, cols in blocks.items():
        log.info("block %s: %d columns", name, len(cols))


if __name__ == "__main__":
    main()
