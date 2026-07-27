"""Stage 0 data gate: verify archive, unpack, validate schema, write parquet.

Two acquisition paths are supported (DECISIONS.md D-002):
  1. `kaggle competitions download -c ieee-fraud-detection -p data/raw/`
     (standard reproduction path, requires Kaggle credentials), or
  2. a pre-placed `data/raw/ieee-fraud-detection.zip` (used in the sandboxed
     build environment, where kaggle.com is unreachable).

Raw data is never committed; only checksums and the validation report are.
"""
from __future__ import annotations

import hashlib
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.utils import get_logger, save_json, timer

log = get_logger("data.prepare")

# Columns that are strings/categoricals in the raw CSVs.
CAT_TRANSACTION = ["ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain",
                   "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9"]
CAT_IDENTITY = ["id_12", "id_15", "id_16", "id_23", "id_27", "id_28", "id_29",
                "id_30", "id_31", "id_33", "id_34", "id_35", "id_36", "id_37",
                "id_38", "DeviceType", "DeviceInfo"]


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def locate_archive() -> Path | None:
    candidates = [config.DATA_RAW / config.ZIP_NAME,
                  Path("/mnt/user-data/uploads/claude-fraud-kit") / config.ZIP_NAME]
    for c in candidates:
        if c.exists():
            return c
    return None


def unpack(archive: Path) -> None:
    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
        missing = [f for f in config.EXPECTED_FILES if f not in names]
        if missing:
            raise SystemExit(f"Archive is missing expected members: {missing}")
        for member in config.EXPECTED_FILES:
            target = config.DATA_RAW / member
            if not target.exists():
                log.info("extracting %s", member)
                zf.extract(member, config.DATA_RAW)


def _dtype_map(csv_path: Path, cat_cols: list[str]) -> dict:
    header = pd.read_csv(csv_path, nrows=0)
    dtypes: dict[str, object] = {}
    for col in header.columns:
        if col == "TransactionID":
            dtypes[col] = "int64"
        elif col == "isFraud":
            dtypes[col] = "int8"
        elif col == "TransactionDT":
            dtypes[col] = "int64"
        elif col in cat_cols:
            dtypes[col] = "object"
        else:
            dtypes[col] = "float32"
    return dtypes


def _read_csv(path: Path, cat_cols: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=_dtype_map(path, cat_cols))
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


def add_time_index(df: pd.DataFrame) -> pd.DataFrame:
    """day / week / month per the fixed temporal protocol (config)."""
    day = (df["TransactionDT"] // config.DAY_SECONDS).astype("int32")
    df["day"] = day
    df["week"] = (day // config.WEEK_DAYS).astype("int32")
    df["month"] = np.minimum(day // config.DAYS_PER_MONTH,
                             config.N_MONTHS - 1).astype("int8")
    return df


def build_train_parquet() -> dict:
    with timer("read train_transaction", log):
        txn = _read_csv(config.DATA_RAW / config.TRAIN_TRANSACTION, CAT_TRANSACTION)
    with timer("read train_identity", log):
        idn = _read_csv(config.DATA_RAW / config.TRAIN_IDENTITY, CAT_IDENTITY)

    n_txn, n_idn = len(txn), len(idn)
    with timer("join identity", log):
        df = txn.merge(idn, on="TransactionID", how="left")
        df["has_identity"] = df["TransactionID"].isin(idn["TransactionID"]).astype("int8")
    del txn, idn

    df = add_time_index(df)
    df = df.sort_values("TransactionDT", kind="stable").reset_index(drop=True)

    out = config.DATA_PROCESSED / "train.parquet"
    with timer("write train.parquet", log):
        df.to_parquet(out, index=False)

    stats = {
        "rows_transaction": n_txn,
        "rows_identity": n_idn,
        "rows_joined": len(df),
        "n_columns": df.shape[1],
        "identity_coverage": float(df["has_identity"].mean()),
        "fraud_rate": float(df["isFraud"].mean()),
        "n_fraud": int(df["isFraud"].sum()),
        "transactiondt_min": int(df["TransactionDT"].min()),
        "transactiondt_max": int(df["TransactionDT"].max()),
        "day_min": int(df["day"].min()),
        "day_max": int(df["day"].max()),
        "span_days": int(df["day"].max() - df["day"].min() + 1),
        "rows_per_month": {int(k): int(v) for k, v in
                           df["month"].value_counts().sort_index().items()},
        "fraud_rate_per_month": {int(k): float(v) for k, v in
                                 df.groupby("month", observed=True)["isFraud"].mean().items()},
        "transaction_amt_min": float(df["TransactionAmt"].min()),
        "transaction_amt_max": float(df["TransactionAmt"].max()),
        "transaction_amt_mean": float(df["TransactionAmt"].mean()),
    }
    return stats


def build_test_parquet() -> dict:
    """Kaggle's unlabeled test set. Converted only for the adversarial-validation
    drift demonstration (clearly labeled as such); never used for evaluation."""
    with timer("read test_transaction", log):
        txn = _read_csv(config.DATA_RAW / config.TEST_TRANSACTION, CAT_TRANSACTION)
    # identity join is not needed for adversarial validation on transaction
    # features; keep the file small.
    txn = add_time_index(txn)
    out = config.DATA_PROCESSED / "test_transaction.parquet"
    with timer("write test_transaction.parquet", log):
        txn.to_parquet(out, index=False)
    stats = {"rows": len(txn), "n_columns": txn.shape[1],
             "day_min": int(txn["day"].min()), "day_max": int(txn["day"].max())}
    del txn
    return stats


def validate(train_stats: dict) -> list[str]:
    """Hard assertions on the schema facts the paper depends on."""
    problems = []
    if not (550_000 <= train_stats["rows_joined"] <= 650_000):
        problems.append(f"unexpected row count {train_stats['rows_joined']}")
    if not (390 <= train_stats["n_columns"] <= 450):
        problems.append(f"unexpected column count {train_stats['n_columns']}")
    if not (0.02 <= train_stats["fraud_rate"] <= 0.06):
        problems.append(f"unexpected fraud rate {train_stats['fraud_rate']:.4f}")
    if not (170 <= train_stats["span_days"] <= 190):
        problems.append(f"unexpected time span {train_stats['span_days']} days")
    months = set(train_stats["rows_per_month"])
    if months != set(range(config.N_MONTHS)):
        problems.append(f"unexpected month indices {sorted(months)}")
    return problems


def main() -> None:
    archive = locate_archive()
    if archive is None:
        raise SystemExit(
            "ieee-fraud-detection.zip not found. Either run\n"
            "  kaggle competitions download -c ieee-fraud-detection -p data/raw/\n"
            "or place the archive at data/raw/ieee-fraud-detection.zip")

    log.info("archive: %s (%.1f MB)", archive, archive.stat().st_size / 1e6)
    checksums = {"archive": {"path": str(archive), "sha256": sha256(archive),
                             "bytes": archive.stat().st_size}}
    unpack(archive)
    for member in config.EXPECTED_FILES:
        p = config.DATA_RAW / member
        checksums[member] = {"sha256": sha256(p), "bytes": p.stat().st_size}
    save_json(checksums, "data_checksums")

    train_stats = build_train_parquet()
    problems = validate(train_stats)
    test_stats = build_test_parquet()

    report = {"train": train_stats, "kaggle_unlabeled_test": test_stats,
              "validation_problems": problems, "status": "FAIL" if problems else "PASS"}
    save_json(report, "data_validation")

    print("\n=== schema validation report ===")
    for k, v in train_stats.items():
        print(f"  {k}: {v}")
    print(f"  status: {report['status']}")
    if problems:
        for p in problems:
            print(f"  PROBLEM: {p}")
        sys.exit(1)


if __name__ == "__main__":
    main()
