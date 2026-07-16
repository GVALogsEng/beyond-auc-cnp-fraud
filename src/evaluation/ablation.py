"""Stage 6: feature-block ablation (incremental validity).

Which data earns its complexity? Forward addition A -> F plus
leave-one-block-out, all evaluated on the month-4 validation set under the
uniform holdout-calibration protocol (same as Stage 5), with fixed tuned
hyperparameters and a fixed tree count so that only the feature set varies.

The final configuration (full set, or full minus a strictly-harmful block)
is confirmed exactly once on the test month by final_test.py.
"""
from __future__ import annotations

import gc

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src import config
from src.data.load import load_features
from src.evaluation.calibration import IsotonicCalibrator, PlattCalibrator
from src.evaluation.cost import baseline_cost, cost_threshold, total_cost
from src.evaluation.metrics import tpr_at_fpr
from src.models import lgbm as L
from src.utils import get_logger, load_json, save_json, timer

log = get_logger("evaluation.ablation")

BLOCK_ORDER = ["A_core", "B_counts", "C_timedeltas", "D_matches",
               "E_vesta", "F_identity"]


def _configs() -> dict[str, list[str]]:
    configs: dict[str, list[str]] = {}
    for i in range(1, len(BLOCK_ORDER) + 1):
        blocks = BLOCK_ORDER[:i]
        name = "fwd_" + "+".join(b.split("_")[0] for b in blocks)
        configs[name] = blocks
    for b in BLOCK_ORDER[:-1]:
        configs[f"lobo_minus_{b.split('_')[0]}"] = \
            [x for x in BLOCK_ORDER if x != b]
    # lobo_minus_F == fwd_A+B+C+D+E; full == fwd_A..F -- dedupe via key on sets
    return configs


def _evaluate_config(df, cols, params, n_estimators, calib_method):
    tr_all = df["month"].isin(config.TRAIN_MONTHS).to_numpy()
    days = df.loc[tr_all, "day"]
    cutoff = int(days.max()) - config.CALIB_HOLDOUT_DAYS + 1
    tr = tr_all & (df["day"] < cutoff).to_numpy()
    cal = tr_all & (df["day"] >= cutoff).to_numpy()
    val = (df["month"] == config.VAL_MONTH).to_numpy()

    y_tr = df.loc[tr, "isFraud"].to_numpy()
    model = L.make_model(params, L.pos_weight(y_tr), n_estimators)
    model.fit(df.loc[tr, cols], y_tr, callbacks=[lgb.log_evaluation(0)])

    raw_cal = model.predict_proba(df.loc[cal, cols], raw_score=True)
    calib = (PlattCalibrator() if calib_method == "platt"
             else IsotonicCalibrator()).fit(raw_cal, df.loc[cal, "isFraud"].to_numpy())

    raw_val = model.predict_proba(df.loc[val, cols], raw_score=True)
    p_val = calib.predict(raw_val)
    y_val = df.loc[val, "isFraud"].to_numpy()
    amt = df.loc[val, "TransactionAmt"].to_numpy(dtype="float64")

    thr = cost_threshold(config.K_CENTRAL)
    base = baseline_cost(y_val, amt, config.K_CENTRAL)["baseline"]
    cost = total_cost(y_val, amt, p_val > thr, config.K_CENTRAL)
    out = {"n_features": len(cols),
           "tpr_at_5fpr": tpr_at_fpr(y_val, raw_val),
           "pr_auc": float(average_precision_score(y_val, raw_val)),
           "savings": 1.0 - cost / base}
    del model
    gc.collect()
    return out


def main() -> None:
    df = load_features()
    blocks = load_json("feature_blocks")["blocks"]
    params = load_json("stage2_lgbm_tuning")["best_params"]
    n_estimators = load_json("stage2_lgbm_regimes")["final_n_estimators"]
    calib_method = load_json("stage3_calibration")["chosen_method"]

    configs = _configs()
    cache: dict[frozenset, dict] = {}
    results: dict[str, dict] = {}
    for name, blist in configs.items():
        cols = list(dict.fromkeys(c for b in blist for c in blocks[b]))
        key = frozenset(cols)
        if key not in cache:
            with timer(f"ablation config {name} ({len(cols)} features)", log):
                cache[key] = _evaluate_config(df, cols, params, n_estimators,
                                              calib_method)
        results[name] = dict(cache[key], blocks=blist)

    full_name = "fwd_" + "+".join(b.split("_")[0] for b in BLOCK_ORDER)
    full = results[full_name]
    for name, res in results.items():
        res["delta_tpr_at_5fpr"] = res["tpr_at_5fpr"] - full["tpr_at_5fpr"]
        res["delta_savings"] = res["savings"] - full["savings"]

    # forward-addition marginal contribution of each block
    fwd_names = [n for n in results if n.startswith("fwd_")]
    fwd_names.sort(key=lambda n: results[n]["n_features"])
    marginal = {}
    prev = None
    for i, n in enumerate(fwd_names):
        b = BLOCK_ORDER[i]
        marginal[b] = {
            "tpr_at_5fpr": results[n]["tpr_at_5fpr"] - (prev["tpr_at_5fpr"] if prev else 0.0),
            "savings": results[n]["savings"] - (prev["savings"] if prev else 0.0)}
        prev = results[n]

    # selection: drop a block only if removing it improves BOTH val metrics
    best_drop, best_gain = None, 0.0
    for b in BLOCK_ORDER:
        name = f"lobo_minus_{b.split('_')[0]}"
        if name not in results:
            continue
        r = results[name]
        if r["delta_tpr_at_5fpr"] > 0 and r["delta_savings"] > 0 \
                and r["delta_savings"] > best_gain:
            best_drop, best_gain = b, r["delta_savings"]
    selected_blocks = ([b for b in BLOCK_ORDER if b != best_drop]
                       if best_drop else list(BLOCK_ORDER))

    payload = {"results": results,
               "forward_marginal": marginal,
               "full_config": full_name,
               "dropped_block": best_drop,
               "selected_blocks": selected_blocks,
               "protocol": ("train months 0-3 minus last "
                            f"{config.CALIB_HOLDOUT_DAYS}d; calibrate on holdout; "
                            "evaluate on month-4 validation; fixed params/trees")}
    save_json(payload, "stage6_ablation")
    log.info("selected blocks: %s (dropped: %s)", selected_blocks, best_drop)


if __name__ == "__main__":
    main()
