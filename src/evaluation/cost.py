"""Stage 4: the cost-sensitive decision layer.

Cost model (per transaction of amount A):
  * approve a fraud   (FN): cost = A            (fraud write-off proxy)
  * decline a legit   (FP): cost = k * A        (lost margin + attrition proxy)
  * approve legit / decline fraud: cost = 0

Expected-cost decisioning declines iff  p*A > (1-p)*k*A,  i.e.

    p > k / (1 + k)        (Elkan 2001; amount-independent threshold)

which is only meaningful when p is a calibrated probability -- hence the
uncalibrated-vs-calibrated policy comparison, priced in dollars.

Savings convention (Bahnsen et al.):
    savings = 1 - Cost_policy / Cost_baseline,
    Cost_baseline = min(cost(approve-all), cost(decline-all)).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from src import config
from src.utils import get_logger, load_json, save_json

log = get_logger("evaluation.cost")


def cost_threshold(k: float) -> float:
    return k / (1.0 + k)


def total_cost(y: np.ndarray, amt: np.ndarray, declined: np.ndarray, k: float) -> float:
    fn_cost = float(amt[(y == 1) & (~declined)].sum())
    fp_cost = float(k * amt[(y == 0) & declined].sum())
    return fn_cost + fp_cost


def baseline_cost(y: np.ndarray, amt: np.ndarray, k: float) -> dict:
    approve_all = float(amt[y == 1].sum())
    decline_all = float(k * amt[y == 0].sum())
    return {"approve_all": approve_all, "decline_all": decline_all,
            "baseline": min(approve_all, decline_all),
            "baseline_policy": "approve_all" if approve_all <= decline_all
                               else "decline_all"}


def evaluate_policy(y: np.ndarray, amt: np.ndarray, declined: np.ndarray,
                    k: float) -> dict:
    base = baseline_cost(y, amt, k)
    cost = total_cost(y, amt, declined, k)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    tp = int(((y == 1) & declined).sum())
    fp = int(((y == 0) & declined).sum())
    return {
        "total_cost": cost,
        "savings": 1.0 - cost / base["baseline"],
        "fraud_dollars_caught": float(amt[(y == 1) & declined].sum()),
        "fraud_dollars_missed": float(amt[(y == 1) & ~declined].sum()),
        "legit_dollars_declined": float(amt[(y == 0) & declined].sum()),
        "tpr": tp / max(n_pos, 1),
        "fpr": fp / max(n_neg, 1),
        "precision": tp / max(tp + fp, 1),
        "n_declined": int(declined.sum()),
        "declined_rate": float(declined.mean()),
    }


def select_thresholds_on_val(y_val: np.ndarray, p_uncal: np.ndarray,
                             p_cal: np.ndarray) -> dict:
    """Score-based thresholds selected on validation, frozen for test."""
    prec, rec, thr = precision_recall_curve(y_val, p_cal)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    f1_thr = float(thr[max(int(np.nanargmax(f1[:-1])), 0)])

    fpr, tpr, roc_thr = roc_curve(y_val, p_cal)
    youden_thr = float(roc_thr[int(np.argmax(tpr - fpr))])

    return {"f1_opt": f1_thr, "youden": youden_thr,
            "note": "f1/youden selected on calibrated month-4 scores"}


def policy_declines(policy: str, p_uncal: np.ndarray, p_cal: np.ndarray,
                    thresholds: dict, k: float) -> np.ndarray:
    n = len(p_cal)
    if policy == "approve_all":
        return np.zeros(n, dtype=bool)
    if policy == "decline_all":
        return np.ones(n, dtype=bool)
    if policy == "f1_opt":
        return p_cal > thresholds["f1_opt"]
    if policy == "youden":
        return p_cal > thresholds["youden"]
    if policy == "cost_uncal":
        return p_uncal > cost_threshold(k)
    if policy == "cost_cal":
        return p_cal > cost_threshold(k)
    raise ValueError(policy)


POLICIES = ("approve_all", "decline_all", "f1_opt", "youden",
            "cost_uncal", "cost_cal")


def policy_table(y: np.ndarray, amt: np.ndarray, p_uncal: np.ndarray,
                 p_cal: np.ndarray, thresholds: dict, k: float) -> dict:
    out = {}
    for pol in POLICIES:
        declined = policy_declines(pol, p_uncal, p_cal, thresholds, k)
        out[pol] = evaluate_policy(y, amt, declined, k)
    out["_baseline"] = baseline_cost(y, amt, k)
    return out


def sensitivity_grid(y, amt, p_uncal, p_cal, thresholds) -> dict:
    grid = {}
    for k in config.K_GRID:
        grid[str(k)] = {pol: evaluate_policy(
            y, amt, policy_declines(pol, p_uncal, p_cal, thresholds, k), k)["savings"]
            for pol in POLICIES}
    return grid


def threshold_sweep(y, amt, p, k, n_points: int = 200) -> dict:
    """Cost and savings as a function of the decline threshold."""
    base = baseline_cost(y, amt, k)["baseline"]
    thrs = np.unique(np.concatenate([
        np.linspace(0, 1, n_points), [cost_threshold(k)]]))
    costs = [total_cost(y, amt, p > t, k) for t in thrs]
    return {"thresholds": thrs.tolist(),
            "costs": costs,
            "savings": [1 - c / base for c in costs]}


def main() -> None:
    """Validation-side stage 4: select thresholds, sanity policy table on val.

    The headline (test) table is produced exactly once by final_test.py.
    """
    scores = pd.read_parquet(config.DATA_PROCESSED / "scores.parquet")
    val = (scores["month"] == config.VAL_MONTH).to_numpy()
    y = scores.loc[val, "y"].to_numpy()
    amt = scores.loc[val, "amt"].to_numpy(dtype="float64")
    p_uncal = scores.loc[val, "p_uncal"].to_numpy(dtype="float64")
    p_cal = scores.loc[val, "p_cal"].to_numpy(dtype="float64")

    thresholds = select_thresholds_on_val(y, p_uncal, p_cal)
    thresholds["cost_threshold_central"] = cost_threshold(config.K_CENTRAL)
    save_json(thresholds, "stage4_thresholds")

    table = policy_table(y, amt, p_uncal, p_cal, thresholds, config.K_CENTRAL)
    payload = {
        "k_central": config.K_CENTRAL,
        "policy_table_val": table,
        "sensitivity_val": sensitivity_grid(y, amt, p_uncal, p_cal, thresholds),
        "sweep_cal_val": threshold_sweep(y, amt, p_cal, config.K_CENTRAL),
        "sweep_uncal_val": threshold_sweep(y, amt, p_uncal, config.K_CENTRAL),
    }
    save_json(payload, "stage4_policies_val")
    log.info("val savings (cost_cal, k=%.2f): %.4f",
             config.K_CENTRAL, table["cost_cal"]["savings"])
    calibration_gap = (table["cost_uncal"]["total_cost"]
                       - table["cost_cal"]["total_cost"])
    log.info("val calibration dollar gap: $%s", f"{calibration_gap:,.0f}")


if __name__ == "__main__":
    main()
