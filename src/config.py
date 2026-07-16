"""Central configuration: paths, seeds, and the fixed temporal/cost protocol.

Every number that governs the evaluation protocol lives here so that the
paper, the tests, and the pipeline cannot silently disagree.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------- paths ----
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results" / "metrics"
FIGURES = ROOT / "reports" / "figures"
MODELS = ROOT / "models"
APP_ARTIFACTS = ROOT / "app" / "artifacts"
LOGS = ROOT / "logs"

for _p in (DATA_RAW, DATA_INTERIM, DATA_PROCESSED, RESULTS, FIGURES, MODELS,
           APP_ARTIFACTS, LOGS):
    _p.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------ determinism ----
SEED = 42

# -------------------------------------------------------- temporal protocol ----
# day = TransactionDT // 86400 ; month = min(day // 30, 5)  (trailing 2-3 day
# stub of the 182-day window is merged into month 5 -- see DECISIONS.md D-003)
DAY_SECONDS = 86_400
DAYS_PER_MONTH = 30
N_MONTHS = 6
TRAIN_MONTHS = (0, 1, 2, 3)
VAL_MONTH = 4
TEST_MONTH = 5
WEEK_DAYS = 7

# Drift-stage retraining protocol: every retrained model holds out the last
# CALIB_HOLDOUT_DAYS days of its training window for calibration (uniform
# across static / expanding / sliding policies).
CALIB_HOLDOUT_DAYS = 14
SLIDING_WINDOW_MONTHS = 3

# ------------------------------------------------------------- cost model ----
# FN (missed fraud)  costs TransactionAmt.
# FP (false decline) costs K * TransactionAmt.
# Expected-cost decisioning declines iff p > k / (1 + k)  (Elkan 2001).
K_CENTRAL = 0.15
K_GRID = (0.05, 0.15, 0.30, 0.60, 1.0)

# ------------------------------------------------------------- evaluation ----
FPR_OPERATING = 0.05          # TPR @ 5% FPR operating-point convention
N_BOOTSTRAP = 1000
BOOTSTRAP_CI = 0.95
N_CALIBRATION_BINS = 15

# ------------------------------------------------------------------ data ----
ZIP_NAME = "ieee-fraud-detection.zip"
TRAIN_TRANSACTION = "train_transaction.csv"
TRAIN_IDENTITY = "train_identity.csv"
TEST_TRANSACTION = "test_transaction.csv"   # unlabeled; adversarial-validation demo only
TEST_IDENTITY = "test_identity.csv"
EXPECTED_FILES = (TRAIN_TRANSACTION, TRAIN_IDENTITY, TEST_TRANSACTION,
                  TEST_IDENTITY, "sample_submission.csv")

# ---------------------------------------------------------------- tuning ----
TUNING_MAX_TRIALS = 20        # hard cap (spec allows <= ~30); adaptive below
TUNING_TIME_BUDGET_S = 45 * 60
TUNING_ROW_FRACTION = 0.4     # stratified row subsample used during search only
MAX_ESTIMATORS = 2000
EARLY_STOPPING_ROUNDS = 75

# --------------------------------------------------------------- app demo ----
APP_SAMPLE_ROWS = 20_000
