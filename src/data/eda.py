"""Stage 1 EDA: persist every number/curve the EDA figures and notebook use.

Nothing here influences modeling; it documents the dataset and the leakage
audit inputs (temporal structure, missingness by block, identity coverage).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.data.load import load_train
from src.utils import get_logger, save_json, timer

log = get_logger("data.eda")


def amount_histogram(amt: pd.Series, y: pd.Series) -> dict:
    edges = np.logspace(np.log10(max(amt.min(), 0.1)),
                        np.log10(amt.max()), 60)
    out = {"edges": edges.tolist()}
    for cls, name in ((0, "legit"), (1, "fraud")):
        h, _ = np.histogram(amt[y == cls], bins=edges, density=True)
        out[name] = h.tolist()
    out["quantiles"] = {
        name: {str(q): float(amt[y == cls].quantile(q))
               for q in (0.25, 0.5, 0.75, 0.9, 0.99)}
        for cls, name in ((0, "legit"), (1, "fraud"))}
    return out


def main() -> None:
    with timer("load train.parquet", log):
        df = load_train()

    weekly = df.groupby("week").agg(
        n=("isFraud", "size"), fraud_rate=("isFraud", "mean"),
        fraud_amt=("TransactionAmt", lambda s: 0.0),  # placeholder, replaced below
    ).reset_index()
    fraud_amt = df[df["isFraud"] == 1].groupby("week")["TransactionAmt"].sum()
    weekly["fraud_amt"] = weekly["week"].map(fraud_amt).fillna(0.0)

    blocks = {
        "A_core": ["TransactionAmt", "ProductCD", "card1", "card2", "card3",
                   "card4", "card5", "card6", "addr1", "addr2", "dist1",
                   "dist2", "P_emaildomain", "R_emaildomain"],
        "B_counts": [f"C{i}" for i in range(1, 15)],
        "C_timedeltas": [f"D{i}" for i in range(1, 16)],
        "D_matches": [f"M{i}" for i in range(1, 10)],
        "E_vesta": [f"V{i}" for i in range(1, 340)],
        "F_identity": [c for c in df.columns if c.startswith("id_")]
                      + ["DeviceType", "DeviceInfo"],
    }
    missingness = {b: float(df[cols].isna().to_numpy().mean())
                   for b, cols in blocks.items() if cols}

    monthly = df.groupby("month", observed=True).agg(
        n=("isFraud", "size"), fraud_rate=("isFraud", "mean")).reset_index()

    payload = {
        "n_rows": len(df),
        "n_cols": df.shape[1],
        "fraud_rate": float(df["isFraud"].mean()),
        "n_fraud": int(df["isFraud"].sum()),
        "identity_coverage": float(df["has_identity"].mean()),
        "span_days": int(df["day"].max() - df["day"].min() + 1),
        "weekly": weekly.to_dict(orient="list"),
        "monthly": monthly.to_dict(orient="list"),
        "amount_hist": amount_histogram(df["TransactionAmt"], df["isFraud"]),
        "missingness_by_block": missingness,
        "product_mix": {str(k): int(v) for k, v in
                        df["ProductCD"].value_counts().items()},
        "fraud_rate_by_product": {str(k): float(v) for k, v in
                                  df.groupby("ProductCD", observed=True)["isFraud"]
                                    .mean().items()},
    }
    save_json(payload, "stage1_eda")
    log.info("EDA persisted: %d rows, fraud rate %.4f, identity coverage %.3f",
             len(df), payload["fraud_rate"], payload["identity_coverage"])


if __name__ == "__main__":
    main()
