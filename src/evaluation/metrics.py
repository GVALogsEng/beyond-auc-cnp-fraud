"""Rank metrics and bootstrap confidence intervals."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from src import config


def tpr_at_fpr(y: np.ndarray, score: np.ndarray,
               fpr_target: float = config.FPR_OPERATING) -> float:
    """TPR at a fixed FPR operating point (linear interpolation on the ROC)."""
    fpr, tpr, _ = roc_curve(y, score)
    return float(np.interp(fpr_target, fpr, tpr))


def threshold_at_fpr(y: np.ndarray, score: np.ndarray,
                     fpr_target: float = config.FPR_OPERATING) -> float:
    fpr, _, thr = roc_curve(y, score)
    idx = int(np.searchsorted(fpr, fpr_target, side="right") - 1)
    return float(thr[max(idx, 0)])


def rank_metrics(y: np.ndarray, score: np.ndarray) -> dict:
    return {
        "roc_auc": float(roc_auc_score(y, score)),
        "pr_auc": float(average_precision_score(y, score)),
        "tpr_at_5fpr": tpr_at_fpr(y, score),
    }


def bootstrap_ci(stat_fn, n_rows: int, n_boot: int = config.N_BOOTSTRAP,
                 seed: int = config.SEED, ci: float = config.BOOTSTRAP_CI) -> dict:
    """Percentile bootstrap over row resamples.

    `stat_fn(idx)` computes the statistic on the resampled row indices and
    may return a scalar or a dict of scalars.
    """
    rng = np.random.RandomState(seed)
    draws: list = []
    for _ in range(n_boot):
        idx = rng.randint(0, n_rows, n_rows)
        draws.append(stat_fn(idx))
    lo, hi = 100 * (1 - ci) / 2, 100 * (1 + ci) / 2
    if isinstance(draws[0], dict):
        out = {}
        for key in draws[0]:
            vals = np.array([d[key] for d in draws], dtype="float64")
            vals = vals[~np.isnan(vals)]
            out[key] = {"lo": float(np.percentile(vals, lo)),
                        "hi": float(np.percentile(vals, hi)),
                        "se": float(vals.std())}
        return out
    vals = np.array(draws, dtype="float64")
    vals = vals[~np.isnan(vals)]
    return {"lo": float(np.percentile(vals, lo)),
            "hi": float(np.percentile(vals, hi)),
            "se": float(vals.std())}
