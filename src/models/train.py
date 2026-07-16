"""Stage 2: champion/challenger training under three split regimes.

Regimes (identical features and hyperparameters; only the split changes):
  random   -- stratified shuffled 5-fold CV within the training months.
              This is the protocol a leaderboard-minded practitioner uses,
              and it is *invalid* here (temporal structure).
  grouped  -- GroupKFold with month as group within the training months
              (the honest cross-validation; also sources early-stopped
              tree counts for the final refit).
  temporal -- train on months 0-3, evaluate on month 4 (validation).
              Month 5 is touched exactly once, in the final test pass.

Outputs: results/metrics/stage2_*.json, models/lgbm_final.txt,
data/interim/scores.parquet (per-row scores reused by all later stages).
"""
from __future__ import annotations

import gc

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

from src import config
from src.data.load import load_features
from src.evaluation.metrics import rank_metrics
from src.models import lgbm as L
from src.models.champion_lr import LRMatrixBuilder, fit_lr, lr_feature_cols
from src.utils import (get_logger, json_exists, load_json, save_json, set_seed,
                       timer)

log = get_logger("models.train")


def feature_frame():
    df = load_features()
    blocks = load_json("feature_blocks")["blocks"]
    feat_cols = [c for cols in blocks.values() for c in cols]
    feat_cols = list(dict.fromkeys(feat_cols))
    return df, blocks, feat_cols


def lgbm_regimes(df: pd.DataFrame, feat_cols: list[str], params: dict) -> dict:
    tr_mask = df["month"].isin(config.TRAIN_MONTHS).to_numpy()
    X = df.loc[tr_mask, feat_cols]
    y = df.loc[tr_mask, "isFraud"].to_numpy()
    months = df.loc[tr_mask, "month"].to_numpy()

    results: dict = {"params": params}

    with timer("LGBM random 5-fold CV", log):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED)
        folds = []
        for tr_i, va_i in skf.split(X, y):
            m = L.fit_with_early_stop(params, X.iloc[tr_i], y[tr_i],
                                      X.iloc[va_i], y[va_i])
            p = m.predict_proba(X.iloc[va_i])[:, 1]
            folds.append(rank_metrics(y[va_i], p))
            del m
            gc.collect()
        results["random_cv"] = _fold_summary(folds)

    with timer("LGBM grouped (month) CV", log):
        gkf = GroupKFold(n_splits=len(np.unique(months)))
        folds, iters = [], []
        for tr_i, va_i in gkf.split(X, y, groups=months):
            m = L.fit_with_early_stop(params, X.iloc[tr_i], y[tr_i],
                                      X.iloc[va_i], y[va_i])
            p = m.predict_proba(X.iloc[va_i])[:, 1]
            folds.append(rank_metrics(y[va_i], p))
            iters.append(m.best_iteration_ or config.MAX_ESTIMATORS)
            del m
            gc.collect()
        results["grouped_cv"] = _fold_summary(folds)
        results["grouped_best_iters"] = iters

    n_final = int(np.median(iters) * 1.1)
    results["final_n_estimators"] = n_final

    with timer(f"LGBM final fit (months 0-3, {n_final} trees)", log):
        final = L.make_model(params, L.pos_weight(y), n_final)
        final.fit(X, y, callbacks=[lgb.log_evaluation(0)])
        final.booster_.save_model(str(config.MODELS / "lgbm_final.txt"))

    with timer("LGBM temporal validation (month 4)", log):
        va_mask = (df["month"] == config.VAL_MONTH).to_numpy()
        p_val = final.predict_proba(df.loc[va_mask, feat_cols])[:, 1]
        results["temporal_val"] = rank_metrics(df.loc[va_mask, "isFraud"].to_numpy(),
                                               p_val)

    with timer("LGBM score full table", log):
        raw = final.predict_proba(df[feat_cols], raw_score=True)
        scores = pd.DataFrame({
            "TransactionID": df["TransactionID"],
            "day": df["day"], "week": df["week"], "month": df["month"],
            "y": df["isFraud"].astype("int8"),
            "amt": df["TransactionAmt"].astype("float64"),
            "lgbm_raw": raw.astype("float32"),
            "lgbm_p": (1.0 / (1.0 + np.exp(-raw))).astype("float32"),
        })
        scores.to_parquet(config.DATA_PROCESSED / "scores.parquet", index=False)

    imp = pd.Series(final.booster_.feature_importance("gain"),
                    index=feat_cols).sort_values(ascending=False)
    results["top30_gain_importance"] = {k: float(v) for k, v in imp.head(30).items()}
    del final
    gc.collect()
    return results


def lr_regimes(df: pd.DataFrame, blocks: dict) -> dict:
    cols = lr_feature_cols(blocks)
    tr_mask = df["month"].isin(config.TRAIN_MONTHS).to_numpy()
    sub = df.loc[tr_mask, cols + ["isFraud", "month"]]
    y = sub["isFraud"].to_numpy()
    months = sub["month"].to_numpy()
    results: dict = {"n_features_raw": len(cols)}

    # --- inner C selection on a temporal inner split (train 0-2 -> month 3)
    with timer("LR inner C selection", log):
        inner_tr = months < 3
        inner_va = months == 3
        best_c, best_ap = None, -1.0
        for C in (0.03, 0.1, 0.3):
            b = LRMatrixBuilder(cols).fit(sub.loc[inner_tr, cols])
            m = fit_lr(b.transform(sub.loc[inner_tr, cols]), y[inner_tr], C)
            ap = average_precision_score(
                y[inner_va], m.predict_proba(b.transform(sub.loc[inner_va, cols]))[:, 1])
            log.info("LR C=%.2f month-3 PR-AUC=%.5f", C, ap)
            if ap > best_ap:
                best_c, best_ap = C, ap
        results["C"] = best_c

    with timer("LR random 5-fold CV", log):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED)
        folds = []
        for tr_i, va_i in skf.split(sub[cols], y):
            b = LRMatrixBuilder(cols).fit(sub[cols].iloc[tr_i])
            m = fit_lr(b.transform(sub[cols].iloc[tr_i]), y[tr_i], best_c)
            p = m.predict_proba(b.transform(sub[cols].iloc[va_i]))[:, 1]
            folds.append(rank_metrics(y[va_i], p))
        results["random_cv"] = _fold_summary(folds)

    with timer("LR grouped (month) CV", log):
        gkf = GroupKFold(n_splits=len(np.unique(months)))
        folds = []
        for tr_i, va_i in gkf.split(sub[cols], y, groups=months):
            b = LRMatrixBuilder(cols).fit(sub[cols].iloc[tr_i])
            m = fit_lr(b.transform(sub[cols].iloc[tr_i]), y[tr_i], best_c)
            p = m.predict_proba(b.transform(sub[cols].iloc[va_i]))[:, 1]
            folds.append(rank_metrics(y[va_i], p))
        results["grouped_cv"] = _fold_summary(folds)

    with timer("LR final fit + temporal validation", log):
        builder = LRMatrixBuilder(cols).fit(sub[cols])
        model = fit_lr(builder.transform(sub[cols]), y, best_c)
        va_mask = (df["month"] == config.VAL_MONTH).to_numpy()
        p_val = model.predict_proba(builder.transform(df.loc[va_mask, cols]))[:, 1]
        results["temporal_val"] = rank_metrics(df.loc[va_mask, "isFraud"].to_numpy(),
                                               p_val)
        # score the full table and append to scores.parquet
        p_all = model.predict_proba(builder.transform(df[cols]))[:, 1]
        scores = pd.read_parquet(config.DATA_PROCESSED / "scores.parquet")
        scores["lr_p"] = p_all.astype("float32")
        scores.to_parquet(config.DATA_PROCESSED / "scores.parquet", index=False)

    del sub
    gc.collect()
    return results


def _fold_summary(folds: list[dict]) -> dict:
    keys = folds[0].keys()
    return {"folds": folds,
            "mean": {k: float(np.mean([f[k] for f in folds])) for k in keys},
            "sd": {k: float(np.std([f[k] for f in folds])) for k in keys}}


def main() -> None:
    set_seed()
    df, blocks, feat_cols = feature_frame()
    log.info("feature table: %d rows x %d features", len(df), len(feat_cols))

    if json_exists("stage2_lgbm_tuning"):
        params = load_json("stage2_lgbm_tuning")["best_params"]
        log.info("reusing cached tuning result")
    else:
        tr_mask = df["month"].isin(config.TRAIN_MONTHS).to_numpy()
        with timer("LGBM tuning", log):
            params = L.tune(df.loc[tr_mask, feat_cols],
                            df.loc[tr_mask, "isFraud"].to_numpy(),
                            df.loc[tr_mask, "month"].to_numpy())

    lgbm_res = lgbm_regimes(df, feat_cols, params)
    save_json(lgbm_res, "stage2_lgbm_regimes")

    lr_res = lr_regimes(df, blocks)
    save_json(lr_res, "stage2_lr_regimes")

    # optimism gap on validation (final test-side gap added in the test pass)
    gap = {
        "lgbm": {"random_cv_mean": lgbm_res["random_cv"]["mean"],
                 "grouped_cv_mean": lgbm_res["grouped_cv"]["mean"],
                 "temporal_val": lgbm_res["temporal_val"]},
        "lr": {"random_cv_mean": lr_res["random_cv"]["mean"],
               "grouped_cv_mean": lr_res["grouped_cv"]["mean"],
               "temporal_val": lr_res["temporal_val"]},
    }
    for mdl in gap:
        gap[mdl]["gap_random_minus_temporal_val"] = {
            k: gap[mdl]["random_cv_mean"][k] - gap[mdl]["temporal_val"][k]
            for k in gap[mdl]["temporal_val"]}
    save_json(gap, "stage2_optimism_gap")

    # model selection for the decision layer (on month-4 validation)
    sel = "lgbm" if (lgbm_res["temporal_val"]["pr_auc"]
                     >= lr_res["temporal_val"]["pr_auc"]) else "lr"
    save_json({"selected": sel,
               "lgbm_val": lgbm_res["temporal_val"],
               "lr_val": lr_res["temporal_val"],
               "criterion": "month-4 validation PR-AUC"},
              "stage2_model_selection")
    log.info("selected model for the decision layer: %s", sel)


if __name__ == "__main__":
    main()
