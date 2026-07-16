"""Champion model: regularized logistic regression.

Design constraints (DECISIONS.md D-005):
  * encodings and statistics (one-hot vocabularies, medians, scaler moments)
    are fit on the *fitting window only* and frozen for any later data;
  * frequency information enters via the causal expanding counts built in
    src.features.build, never via full-table frequencies;
  * V1-V339 are excluded from the champion -- the simple benchmark uses the
    interpretable blocks (A-D, F numeric); the challenger uses everything.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src import config

ONEHOT_COLS = ["ProductCD", "card4", "card6", "M1", "M2", "M3", "M4", "M5",
               "M6", "M7", "M8", "M9", "DeviceType", "P_emaildomain",
               "R_emaildomain"]
ONEHOT_TOPK = 30
INDICATOR_PREFIXES = ("D",)   # missingness indicators for the D block


MAX_ONEHOT_CARDINALITY = 60


class LRMatrixBuilder:
    """Fit-on-train / transform-anywhere design matrix for the champion.

    Columns are partitioned by *dtype at fit time*: numeric columns are
    imputed+scaled; non-numeric columns with modest cardinality are one-hot
    encoded on the fit window's top categories; very-high-cardinality strings
    (e.g. DeviceInfo, id_33) are dropped from the champion.
    """

    def __init__(self, feature_cols: list[str]):
        self.feature_cols = list(feature_cols)
        self.numeric_cols: list[str] = []
        self.onehot_cols: list[str] = []
        self.dropped_cols: list[str] = []
        self.vocab: dict[str, list] = {}
        self.medians: pd.Series | None = None
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.columns_: list[str] | None = None

    def fit(self, df: pd.DataFrame) -> "LRMatrixBuilder":
        for c in self.feature_cols:
            if pd.api.types.is_numeric_dtype(df[c]):
                self.numeric_cols.append(c)
            elif df[c].nunique() <= MAX_ONEHOT_CARDINALITY or c in ONEHOT_COLS:
                self.onehot_cols.append(c)
            else:
                self.dropped_cols.append(c)
        for c in self.onehot_cols:
            top = df[c].astype("str").value_counts().head(ONEHOT_TOPK).index.tolist()
            self.vocab[c] = top
        num = df[self.numeric_cols].astype("float32")
        # columns that are entirely missing in the fit window fall back to 0
        self.medians = num.median().fillna(0.0)
        X = self._assemble(df)
        self.columns_ = list(X.columns)
        arr = X.to_numpy(dtype="float32")
        self.mean_ = arr.mean(axis=0)
        self.scale_ = arr.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def _assemble(self, df: pd.DataFrame) -> pd.DataFrame:
        parts = []
        num = df[self.numeric_cols].astype("float32")
        for pref in INDICATOR_PREFIXES:
            cols = [c for c in self.numeric_cols
                    if c.startswith(pref) and c[len(pref):].isdigit()]
            if cols:
                ind = num[cols].isna().astype("float32")
                ind.columns = [f"{c}_missing" for c in cols]
                parts.append(ind)
        num = num.fillna(self.medians)
        parts.insert(0, num)
        for c in self.onehot_cols:
            s = df[c].astype("str")
            s = s.where(s.isin(self.vocab[c]), "OTHER")
            d = pd.get_dummies(s, prefix=c, dtype="float32")
            for v in self.vocab[c] + ["OTHER"]:
                col = f"{c}_{v}"
                if col not in d.columns:
                    d[col] = np.float32(0.0)
            parts.append(d[[f"{c}_{v}" for v in self.vocab[c] + ["OTHER"]]])
        return pd.concat(parts, axis=1)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        X = self._assemble(df)
        X = X.reindex(columns=self.columns_, fill_value=0.0)
        arr = X.to_numpy(dtype="float32")
        return (arr - self.mean_) / self.scale_


def lr_feature_cols(blocks: dict[str, list[str]]) -> list[str]:
    cols: list[str] = []
    for b in ("A_core", "B_counts", "C_timedeltas", "D_matches", "F_identity"):
        cols += blocks[b]
    return [c for c in cols if c != "DeviceInfo"]


def fit_lr(X: np.ndarray, y: np.ndarray, C: float = 0.1) -> LogisticRegression:
    model = LogisticRegression(C=C, class_weight="balanced", solver="lbfgs",
                               max_iter=1000, tol=1e-4, n_jobs=2,
                               random_state=config.SEED)
    model.fit(X, y)
    return model
