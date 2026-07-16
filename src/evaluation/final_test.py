"""The single strict out-of-time test pass (month 5).

Everything upstream -- hyperparameters, tree counts, feature configuration,
calibration method, decision thresholds -- was frozen using months 0-4 only.
This module touches month 5 exactly once and persists every number the paper
reports about the test set, including bootstrap confidence intervals.

Also exports the demo artifacts (stratified test sample, per-row SHAP top
contributors, narratives) so the Streamlit app has no Kaggle dependency.
"""
from __future__ import annotations

import gc

import lightgbm as lgb
import numpy as np
import pandas as pd

from src import config
from src.data.load import load_features
from src.evaluation import cost as C
from src.evaluation.calibration import (IsotonicCalibrator, PlattCalibrator,
                                        calib_metrics, reliability_bins,
                                        weekly_calibration)
from src.evaluation.metrics import bootstrap_ci, rank_metrics
from src.models import lgbm as L
from src.narratives import narrative
from src.utils import get_logger, load_json, save_json, set_seed, timer

log = get_logger("evaluation.final_test")

DISPLAY_COLS = ["ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain",
                "DeviceType", "hour", "dow", "addr1", "dist1",
                "card1_past_count", "card1_since_last", "card1_amt_ratio",
                "C13", "D1", "has_identity"]


def _maybe_refit_selected(df, blocks, feat_cols_full):
    """If the ablation dropped a block, refit the final model on the selected
    configuration (train months 0-3), recalibrate on month 4, and rewrite the
    score table. Otherwise reuse the stage-2 final model."""
    abl = load_json("stage6_ablation")
    selected_blocks = abl["selected_blocks"]
    cols = list(dict.fromkeys(c for b in selected_blocks for c in blocks[b]))
    if abl["dropped_block"] is None:
        return pd.read_parquet(config.DATA_PROCESSED / "scores.parquet"), cols, False

    log.info("ablation dropped %s -> refitting final model on %d features",
             abl["dropped_block"], len(cols))
    params = load_json("stage2_lgbm_tuning")["best_params"]
    n_final = load_json("stage2_lgbm_regimes")["final_n_estimators"]
    tr = df["month"].isin(config.TRAIN_MONTHS).to_numpy()
    y_tr = df.loc[tr, "isFraud"].to_numpy()
    model = L.make_model(params, L.pos_weight(y_tr), n_final)
    model.fit(df.loc[tr, cols], y_tr, callbacks=[lgb.log_evaluation(0)])
    model.booster_.save_model(str(config.MODELS / "lgbm_final.txt"))

    raw = model.predict_proba(df[cols], raw_score=True)
    scores = pd.read_parquet(config.DATA_PROCESSED / "scores.parquet")
    scores["lgbm_raw"] = raw.astype("float32")
    scores["lgbm_p"] = (1 / (1 + np.exp(-raw))).astype("float32")
    scores["p_uncal"] = scores["lgbm_p"]

    val = (scores["month"] == config.VAL_MONTH).to_numpy()
    method = load_json("stage3_calibration")["chosen_method"]
    calib = (PlattCalibrator() if method == "platt" else IsotonicCalibrator())
    calib.fit(raw[val], scores.loc[val, "y"].to_numpy())
    scores["p_cal"] = calib.predict(raw).astype("float32")
    scores.to_parquet(config.DATA_PROCESSED / "scores.parquet", index=False)
    del model
    gc.collect()
    return scores, cols, True


def evaluate_test(scores: pd.DataFrame) -> dict:
    test = (scores["month"] == config.TEST_MONTH).to_numpy()
    y = scores.loc[test, "y"].to_numpy()
    amt = scores.loc[test, "amt"].to_numpy(dtype="float64")
    raw = scores.loc[test, "lgbm_raw"].to_numpy(dtype="float64")
    p_uncal = scores.loc[test, "p_uncal"].to_numpy(dtype="float64")
    p_cal = scores.loc[test, "p_cal"].to_numpy(dtype="float64")
    thresholds = load_json("stage4_thresholds")

    # ---- rank metrics + final optimism gap --------------------------------
    rank_test = rank_metrics(y, raw)
    lr_rank_test = rank_metrics(y, scores.loc[test, "lr_p"].to_numpy())
    gap_src = load_json("stage2_optimism_gap")
    gap_final = {
        "lgbm": {"random_cv_mean": gap_src["lgbm"]["random_cv_mean"],
                 "grouped_cv_mean": gap_src["lgbm"]["grouped_cv_mean"],
                 "test": rank_test,
                 "gap_random_minus_test": {
                     k: gap_src["lgbm"]["random_cv_mean"][k] - rank_test[k]
                     for k in rank_test}},
        "lr": {"random_cv_mean": gap_src["lr"]["random_cv_mean"],
               "grouped_cv_mean": gap_src["lr"]["grouped_cv_mean"],
               "test": lr_rank_test,
               "gap_random_minus_test": {
                   k: gap_src["lr"]["random_cv_mean"][k] - lr_rank_test[k]
                   for k in lr_rank_test}},
    }
    save_json(gap_final, "stage2_optimism_gap_final")

    # ---- policy table with bootstrap CIs ----------------------------------
    table = C.policy_table(y, amt, p_uncal, p_cal, thresholds, config.K_CENTRAL)

    def stat_fn(idx):
        out = {}
        yb, ab = y[idx], amt[idx]
        if yb.sum() == 0:
            return {k: np.nan for k in
                    [f"savings_{p}" for p in C.POLICIES]
                    + ["tpr_at_5fpr", "pr_auc", "roc_auc"]}
        for pol in C.POLICIES:
            declined = C.policy_declines(pol, p_uncal[idx], p_cal[idx],
                                         thresholds, config.K_CENTRAL)
            out[f"savings_{pol}"] = C.evaluate_policy(yb, ab, declined,
                                                      config.K_CENTRAL)["savings"]
        rm = rank_metrics(yb, raw[idx])
        out.update({"tpr_at_5fpr": rm["tpr_at_5fpr"], "pr_auc": rm["pr_auc"],
                    "roc_auc": rm["roc_auc"]})
        return out

    with timer(f"bootstrap x{config.N_BOOTSTRAP}", log):
        cis = bootstrap_ci(stat_fn, len(y))

    payload = {
        "k_central": config.K_CENTRAL,
        "n_test": int(test.sum()),
        "test_fraud_rate": float(y.mean()),
        "policy_table_test": table,
        "bootstrap_ci": cis,
        "rank_metrics_test": rank_test,
        "sensitivity_test": C.sensitivity_grid(y, amt, p_uncal, p_cal, thresholds),
        "sweep_cal_test": C.threshold_sweep(y, amt, p_cal, config.K_CENTRAL),
        "sweep_uncal_test": C.threshold_sweep(y, amt, p_uncal, config.K_CENTRAL),
        "calibration_dollar_gap": {
            "cost_uncal_minus_cost_cal": (table["cost_uncal"]["total_cost"]
                                          - table["cost_cal"]["total_cost"]),
            "as_pct_of_baseline": ((table["cost_uncal"]["total_cost"]
                                    - table["cost_cal"]["total_cost"])
                                   / table["_baseline"]["baseline"]),
        },
    }
    save_json(payload, "stage4_policies_test")

    # ---- calibration on test ----------------------------------------------
    p_platt = scores.loc[test, "p_platt"].to_numpy(dtype="float64")
    p_iso = scores.loc[test, "p_isotonic"].to_numpy(dtype="float64")
    oot_weeks = np.unique(scores.loc[scores["month"] >= config.VAL_MONTH, "week"])
    calib_test = {
        "test_metrics": {"uncalibrated": calib_metrics(y, p_uncal),
                         "platt": calib_metrics(y, p_platt),
                         "isotonic": calib_metrics(y, p_iso)},
        "reliability_test": {"uncalibrated": reliability_bins(y, p_uncal),
                             "platt": reliability_bins(y, p_platt),
                             "isotonic": reliability_bins(y, p_iso)},
        "weekly_oot_calibration": {
            "uncalibrated": weekly_calibration(scores, "p_uncal", oot_weeks),
            "calibrated": weekly_calibration(scores, "p_cal", oot_weeks)},
    }
    save_json(calib_test, "stage3_calibration_test")
    return payload


def shap_and_app_artifacts(df, scores, feat_cols, table_payload):
    set_seed()
    test_idx = np.flatnonzero((scores["month"] == config.TEST_MONTH).to_numpy())
    y_test = scores["y"].to_numpy()[test_idx]

    rng = np.random.RandomState(config.SEED)
    n = min(config.APP_SAMPLE_ROWS, len(test_idx))
    take = []
    for cls in (0, 1):
        cls_idx = test_idx[y_test == cls]
        k = int(round(n * (len(cls_idx) / len(test_idx))))
        take.append(rng.choice(cls_idx, size=min(k, len(cls_idx)), replace=False))
    sample_idx = np.sort(np.concatenate(take))

    booster = lgb.Booster(model_file=str(config.MODELS / "lgbm_final.txt"))
    Xs = df.iloc[sample_idx][feat_cols]
    with timer(f"SHAP (pred_contrib) on {len(sample_idx)} test rows", log):
        contrib = booster.predict(Xs, pred_contrib=True)
    shap_values, base_value = contrib[:, :-1], float(contrib[0, -1])
    np.savez_compressed(config.DATA_INTERIM / "shap_sample.npz",
                        values=shap_values.astype("float32"),
                        base_value=base_value,
                        feature_names=np.array(feat_cols, dtype=object),
                        row_idx=sample_idx)

    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    save_json({"base_value_logodds": base_value,
               "mean_abs_shap_top30": {feat_cols[i]: float(mean_abs[i])
                                       for i in order[:30]},
               "n_sample": len(sample_idx)},
              "stage7_shap")

    # ---------------- app sample table -------------------------------------
    thr = C.cost_threshold(config.K_CENTRAL)
    meta = scores.iloc[sample_idx][["TransactionID", "day", "week", "month",
                                    "y", "amt", "p_uncal", "p_cal"]].reset_index(drop=True)
    meta["declined_central"] = (meta["p_cal"] > thr)
    disp = df.iloc[sample_idx][[c for c in DISPLAY_COLS if c in df.columns]] \
             .reset_index(drop=True)
    sample = pd.concat([meta, disp], axis=1)
    sample.to_parquet(config.APP_ARTIFACTS / "sample.parquet", index=False)

    # per-row top-10 SHAP contributors (long format, app waterfalls)
    top_k = 10
    rows = []
    raw_vals = Xs.reset_index(drop=True)
    for r in range(len(sample_idx)):
        o = np.argsort(np.abs(shap_values[r]))[::-1][:top_k]
        for rank, j in enumerate(o):
            v = raw_vals.iloc[r, j]
            rows.append((int(meta.loc[r, "TransactionID"]), rank, feat_cols[j],
                         None if pd.isna(v) else str(v),
                         float(shap_values[r, j])))
    pd.DataFrame(rows, columns=["TransactionID", "rank", "feature",
                                "value", "shap"]) \
      .to_parquet(config.APP_ARTIFACTS / "shap_top.parquet", index=False)

    # narratives for declined transactions (cold path)
    declined = sample[sample["declined_central"]].copy()
    declined = declined.sort_values("p_cal", ascending=False).head(400)
    shap_top = pd.read_parquet(config.APP_ARTIFACTS / "shap_top.parquet")
    narratives = []
    for _, row in declined.iterrows():
        contribs = shap_top[shap_top["TransactionID"] == row["TransactionID"]]
        contribs = [{"feature": c.feature,
                     "value": c.value, "shap": float(c.shap)}
                    for c in contribs.itertuples()]
        txn = {"TransactionID": int(row["TransactionID"]),
               "amt": float(row["amt"]), "p_cal": float(row["p_cal"]),
               "ProductCD": str(row.get("ProductCD", "?")),
               "hour": int(row["hour"]) if "hour" in row else None}
        narratives.append(narrative(txn, contribs, thr))
    save_json({"narratives": narratives,
               "n_llm": sum(1 for x in narratives if x["source"] == "anthropic"),
               "n_template": sum(1 for x in narratives if x["source"] == "template")},
              "stage9_narratives")
    import json as _json
    with open(config.APP_ARTIFACTS / "narratives.json", "w") as fh:
        _json.dump(narratives, fh)

    # copies the app needs (kept small, regenerated by `make evaluate`)
    for src_name, dst in [("stage4_policies_test", "policy_table.json"),
                          ("stage5_drift", "drift.json"),
                          ("stage3_calibration_test", "calibration.json"),
                          ("stage2_model_selection", "model_selection.json")]:
        with open(config.APP_ARTIFACTS / dst, "w") as fh:
            _json.dump(load_json(src_name), fh)

    # model card with real numbers
    ci = table_payload["bootstrap_ci"]
    card = {
        "name": "Cost-sensitive calibrated CNP fraud decision system",
        "model": "LightGBM (challenger) + validation-fit calibrator; "
                 "logistic-regression champion benchmarked",
        "data": "IEEE-CIS Fraud Detection (Kaggle/Vesta, 2019 e-commerce); "
                "~590K transactions over ~6 months; labels reflect "
                "chargeback-linked labeling",
        "protocol": "strict out-of-time: train months 0-3, validate month 4, "
                    "test month 5 (touched once)",
        "headline": {
            "savings_at_k_0.15": table_payload["policy_table_test"]["cost_cal"]["savings"],
            "savings_ci95": [ci["savings_cost_cal"]["lo"], ci["savings_cost_cal"]["hi"]],
            "tpr_at_5fpr": table_payload["rank_metrics_test"]["tpr_at_5fpr"],
            "tpr_ci95": [ci["tpr_at_5fpr"]["lo"], ci["tpr_at_5fpr"]["hi"]],
            "pr_auc": table_payload["rank_metrics_test"]["pr_auc"],
        },
        "limitations": [
            "single anonymized 2019 dataset; drift studied is historical",
            "labels are chargeback-linked and inherit reporting noise/delay",
            "cost model is a two-parameter proxy (FN=amount, FP=k*amount)",
            "thresholds assume calibrated probabilities; monitor calibration",
        ],
        "cold_path_llm": "narratives are investigation support only; "
                         "the scoring path is deterministic LightGBM",
    }
    with open(config.APP_ARTIFACTS / "model_card.json", "w") as fh:
        _json.dump(card, fh, indent=2)


def main() -> None:
    set_seed()
    df = load_features()
    blocks = load_json("feature_blocks")["blocks"]
    feat_cols_full = list(dict.fromkeys(c for cols in blocks.values() for c in cols))

    scores, feat_cols, refitted = _maybe_refit_selected(df, blocks, feat_cols_full)
    table_payload = evaluate_test(scores)
    shap_and_app_artifacts(df, scores, feat_cols, table_payload)

    # consolidated headline numbers for the README
    abl = load_json("stage6_ablation")
    drift = load_json("stage5_drift")
    gap = load_json("stage2_optimism_gap_final")
    cal = load_json("stage3_calibration_test")
    m5 = [r for r in drift["weekly"]["static"] if r["month"] == config.TEST_MONTH]
    m5_exp = [r for r in drift["weekly"]["expanding"] if r["month"] == config.TEST_MONTH]
    m5_sld = [r for r in drift["weekly"]["sliding"] if r["month"] == config.TEST_MONTH]
    headline = {
        "final_model_refitted_after_ablation": refitted,
        "selected_blocks": abl["selected_blocks"],
        "test_policy_table": table_payload["policy_table_test"],
        "bootstrap_ci": table_payload["bootstrap_ci"],
        "rank_metrics_test": table_payload["rank_metrics_test"],
        "optimism_gap": gap,
        "calibration_test": cal["test_metrics"],
        "calibration_dollar_gap": table_payload["calibration_dollar_gap"],
        "retraining_month5_mean": {
            "static": {k: float(np.mean([r[k] for r in m5]))
                       for k in ("tpr_at_5fpr", "pr_auc", "savings")},
            "expanding": {k: float(np.mean([r[k] for r in m5_exp]))
                          for k in ("tpr_at_5fpr", "pr_auc", "savings")},
            "sliding": {k: float(np.mean([r[k] for r in m5_sld]))
                        for k in ("tpr_at_5fpr", "pr_auc", "savings")},
        },
    }
    save_json(headline, "stage8_headline")
    log.info("TEST savings (cost_cal, k=%.2f): %.4f  [%.4f, %.4f]",
             config.K_CENTRAL,
             table_payload["policy_table_test"]["cost_cal"]["savings"],
             table_payload["bootstrap_ci"]["savings_cost_cal"]["lo"],
             table_payload["bootstrap_ci"]["savings_cost_cal"]["hi"])


if __name__ == "__main__":
    main()
