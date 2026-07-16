"""Stage 5: temporal decay, retraining cadence, PSI, adversarial validation.

Retraining policies compared under a *uniform* protocol (DECISIONS.md D-006):
every model trains on its window minus the last CALIB_HOLDOUT_DAYS days,
which are held out to fit the (stage-3-chosen) calibrator. Policies:

  static     train once on months 0-3; never retrain.
  expanding  retrain monthly on all past months (month-5 scoring uses 0-4).
  sliding    retrain monthly on the trailing 3 months.

By construction static and expanding coincide until the first retrain point
(month 5); their month-4 curves are identical and plotted as such.
"""
from __future__ import annotations

import gc

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src import config
from src.data.load import load_features
from src.evaluation.calibration import IsotonicCalibrator, PlattCalibrator, brier
from src.evaluation.metrics import rank_metrics, tpr_at_fpr
from src.evaluation.cost import baseline_cost, cost_threshold, total_cost
from src.models import lgbm as L
from src.utils import get_logger, load_json, save_json, timer

log = get_logger("evaluation.drift")

POLICY_WINDOWS = {
    "static":    {config.VAL_MONTH: (0, 1, 2, 3), config.TEST_MONTH: (0, 1, 2, 3)},
    "expanding": {config.VAL_MONTH: (0, 1, 2, 3), config.TEST_MONTH: (0, 1, 2, 3, 4)},
    "sliding":   {config.VAL_MONTH: (1, 2, 3),    config.TEST_MONTH: (2, 3, 4)},
}


def _fit_window(df: pd.DataFrame, feat_cols: list[str], months: tuple,
                params: dict, n_estimators: int, calib_method: str):
    """Fit on window-minus-holdout; calibrate on the holdout."""
    in_window = df["month"].isin(months).to_numpy()
    days = df.loc[in_window, "day"]
    cutoff = int(days.max()) - config.CALIB_HOLDOUT_DAYS + 1
    tr = in_window & (df["day"] < cutoff).to_numpy()
    cal = in_window & (df["day"] >= cutoff).to_numpy()

    y_tr = df.loc[tr, "isFraud"].to_numpy()
    model = L.make_model(params, L.pos_weight(y_tr), n_estimators)
    model.fit(df.loc[tr, feat_cols], y_tr, callbacks=[lgb.log_evaluation(0)])

    raw_cal = model.predict_proba(df.loc[cal, feat_cols], raw_score=True)
    y_cal = df.loc[cal, "isFraud"].to_numpy()
    calib = (PlattCalibrator() if calib_method == "platt"
             else IsotonicCalibrator()).fit(raw_cal, y_cal)
    return model, calib, {"train_rows": int(tr.sum()), "cal_rows": int(cal.sum()),
                          "train_day_max": cutoff - 1, "months": list(months)}


def decay_curves(df: pd.DataFrame, feat_cols: list[str], params: dict,
                 n_estimators: int, calib_method: str) -> dict:
    eval_mask = df["month"].isin([config.VAL_MONTH, config.TEST_MONTH]).to_numpy()
    eval_df = df.loc[eval_mask, ["day", "week", "month", "isFraud",
                                 "TransactionAmt"]].copy()

    # fit each distinct window once
    windows = {w for pol in POLICY_WINDOWS.values() for w in pol.values()}
    fitted: dict[tuple, tuple] = {}
    meta = {}
    for w in sorted(windows):
        with timer(f"retrain window months {list(w)}", log):
            model, calib, info = _fit_window(df, feat_cols, w, params,
                                             n_estimators, calib_method)
            raw = model.predict_proba(df.loc[eval_mask, feat_cols],
                                      raw_score=True)
            fitted[w] = (raw, calib.predict(raw))
            meta[str(list(w))] = info
            del model
            gc.collect()

    # weekly series per policy
    weeks = np.sort(eval_df["week"].unique())
    counts = eval_df.groupby("week").size()
    merge_tail = (len(weeks) > 1 and
                  counts[weeks[-1]] < 0.5 * counts[weeks[:-1]].median())
    if merge_tail:
        log.info("merging stub week %d into week %d", weeks[-1], weeks[-2])
        eval_df.loc[eval_df["week"] == weeks[-1], "week"] = weeks[-2]
        weeks = weeks[:-1]

    thr = cost_threshold(config.K_CENTRAL)
    series: dict = {}
    for pol, win_map in POLICY_WINDOWS.items():
        rows = []
        for w in weeks:
            wk = (eval_df["week"] == w).to_numpy()
            month = int(eval_df.loc[wk, "month"].mode().iloc[0])
            raw, p_cal = fitted[win_map[month]]
            y = eval_df.loc[wk, "isFraud"].to_numpy()
            amt = eval_df.loc[wk, "TransactionAmt"].to_numpy(dtype="float64")
            pc = p_cal[wk]
            base = baseline_cost(y, amt, config.K_CENTRAL)["baseline"]
            cost = total_cost(y, amt, pc > thr, config.K_CENTRAL)
            rows.append({"week": int(w), "month": month, "n": int(wk.sum()),
                         "fraud_rate": float(y.mean()),
                         "tpr_at_5fpr": tpr_at_fpr(y, raw[wk]),
                         "pr_auc": float(average_precision_score(y, raw[wk])),
                         "savings": 1.0 - cost / base,
                         "brier": brier(y, pc)})
        series[pol] = rows
    return {"weekly": series, "windows": meta,
            "calib_method": calib_method,
            "cost_threshold": thr,
            "protocol": ("uniform: train = window minus last "
                         f"{config.CALIB_HOLDOUT_DAYS} days (calibration holdout)")}


# ------------------------------------------------------------------ PSI ----

def _psi(expected: np.ndarray, actual: np.ndarray, eps: float = 1e-4) -> float:
    e = np.clip(expected, eps, None)
    a = np.clip(actual, eps, None)
    e, a = e / e.sum(), a / a.sum()
    return float(np.sum((a - e) * np.log(a / e)))


def psi_feature(train_vals: pd.Series, week_vals: pd.Series,
                is_numeric: bool, bins=None):
    if is_numeric:
        counts_t = np.histogram(train_vals.dropna(), bins=bins)[0].astype(float)
        counts_w = np.histogram(week_vals.dropna(), bins=bins)[0].astype(float)
        counts_t = np.append(counts_t, train_vals.isna().sum())
        counts_w = np.append(counts_w, week_vals.isna().sum())
    else:
        cats = train_vals.astype("str").value_counts().head(20).index
        t = train_vals.astype("str").where(train_vals.astype("str").isin(cats), "OTHER")
        w = week_vals.astype("str").where(week_vals.astype("str").isin(cats), "OTHER")
        idx = list(cats) + ["OTHER"]
        counts_t = t.value_counts().reindex(idx).fillna(0).to_numpy(dtype=float)
        counts_w = w.value_counts().reindex(idx).fillna(0).to_numpy(dtype=float)
    return _psi(counts_t, counts_w)


def psi_over_time(df: pd.DataFrame, features: list[str]) -> dict:
    train_mask = df["month"].isin(config.TRAIN_MONTHS).to_numpy()
    eval_mask = df["month"].isin([config.VAL_MONTH, config.TEST_MONTH]).to_numpy()
    weeks = np.sort(df.loc[eval_mask, "week"].unique())

    out: dict = {"weeks": [int(w) for w in weeks], "psi": {}}
    for feat in features:
        s_train = df.loc[train_mask, feat]
        is_num = pd.api.types.is_numeric_dtype(s_train)
        bins = None
        if is_num:
            qs = np.unique(np.nanquantile(s_train.astype("float64"),
                                          np.linspace(0, 1, 11)))
            if len(qs) < 3:
                continue
            qs[0], qs[-1] = -np.inf, np.inf
            bins = qs
        vals = []
        for w in weeks:
            wk = (df["week"] == w) & eval_mask
            vals.append(psi_feature(s_train, df.loc[wk, feat], is_num, bins))
        out["psi"][feat] = vals
    mat = np.array(list(out["psi"].values()))
    out["mean_psi_per_week"] = mat.mean(axis=0).tolist()
    out["max_psi_per_feature"] = {f: float(np.max(v)) for f, v in out["psi"].items()}
    out["n_features_over_0.2_final_week"] = int(sum(v[-1] > 0.2
                                                    for v in out["psi"].values()))
    return out


# ------------------------------------------- adversarial validation ----

def adversarial(df: pd.DataFrame, feat_cols: list[str], label: np.ndarray,
                tag: str, n_rows: int = 300_000) -> dict:
    rng = np.random.RandomState(config.SEED)
    idx = rng.choice(len(df), size=min(n_rows, len(df)), replace=False)
    X = df.iloc[idx][feat_cols]
    z = label[idx]

    params = dict(num_leaves=31, learning_rate=0.1, n_estimators=150,
                  feature_fraction=0.8)
    aucs, imps = [], np.zeros(len(feat_cols))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED)
    for tr_i, va_i in skf.split(X, z):
        m = lgb.LGBMClassifier(**L.BASE_PARAMS, **params)
        m.fit(X.iloc[tr_i], z[tr_i], callbacks=[lgb.log_evaluation(0)])
        aucs.append(roc_auc_score(z[va_i], m.predict_proba(X.iloc[va_i])[:, 1]))
        imps += m.booster_.feature_importance("gain")
        del m
        gc.collect()
    top = pd.Series(imps, index=feat_cols).sort_values(ascending=False).head(20)
    return {"tag": tag, "cv_auc_mean": float(np.mean(aucs)),
            "cv_auc_sd": float(np.std(aucs)),
            "top20_features": {k: float(v) for k, v in top.items()}}


def main() -> None:
    df = load_features()
    blocks = load_json("feature_blocks")["blocks"]
    feat_cols = list(dict.fromkeys(c for cols in blocks.values() for c in cols))
    engineered = set(load_json("feature_blocks")["engineered_causal"])

    params = load_json("stage2_lgbm_tuning")["best_params"]
    n_final = load_json("stage2_lgbm_regimes")["final_n_estimators"]
    calib_method = load_json("stage3_calibration")["chosen_method"]

    with timer("decay curves (3 retraining policies)", log):
        decay = decay_curves(df, feat_cols, params, n_final, calib_method)
    save_json(decay, "stage5_drift")

    top20 = list(load_json("stage2_lgbm_regimes")["top30_gain_importance"])[:20]
    with timer("PSI top-20 features", log):
        psi = psi_over_time(df, top20)
    save_json(psi, "stage5_psi")

    with timer("adversarial validation", log):
        oot_label = df["month"].isin([config.VAL_MONTH, config.TEST_MONTH]) \
                      .to_numpy().astype(int)
        adv_full = adversarial(df, feat_cols, oot_label, "train_vs_oot_full_features")
        no_agg = [c for c in feat_cols if c not in engineered]
        adv_noagg = adversarial(df, no_agg, oot_label,
                                "train_vs_oot_excl_causal_aggregates")
    save_json({"internal_oot": [adv_full, adv_noagg]}, "stage5_adversarial")

    log.info("adversarial AUC (full): %.4f | (excl aggregates): %.4f",
             adv_full["cv_auc_mean"], adv_noagg["cv_auc_mean"])


if __name__ == "__main__":
    main()
