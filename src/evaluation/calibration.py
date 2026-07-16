"""Stage 3: probability calibration fit on the validation month only.

Both calibrators are fit on month 4 (never on training or test data):
  * Platt scaling  -- 1-D logistic regression on the raw model score;
  * isotonic       -- monotone step function on the raw model score.
The method with the lower validation Brier score is selected and applied,
frozen, to the test month in the final test pass.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from src import config
from src.utils import get_logger, load_json, save_json

log = get_logger("evaluation.calibration")

EPS = 1e-7


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def log_loss_(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = config.N_CALIBRATION_BINS) -> float:
    """Expected calibration error with equal-width bins."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    total = len(y)
    err = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        err += (m.sum() / total) * abs(y[m].mean() - p[m].mean())
    return float(err)


def reliability_bins(y: np.ndarray, p: np.ndarray,
                     n_bins: int = config.N_CALIBRATION_BINS) -> dict:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    out = {"bin_mid": [], "mean_pred": [], "frac_pos": [], "count": []}
    for b in range(n_bins):
        m = idx == b
        out["bin_mid"].append(float((bins[b] + bins[b + 1]) / 2))
        out["count"].append(int(m.sum()))
        if m.sum():
            out["mean_pred"].append(float(p[m].mean()))
            out["frac_pos"].append(float(y[m].mean()))
        else:
            out["mean_pred"].append(None)
            out["frac_pos"].append(None)
    return out


class PlattCalibrator:
    def __init__(self):
        self._lr = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)

    def fit(self, raw: np.ndarray, y: np.ndarray):
        self._lr.fit(raw.reshape(-1, 1), y)
        return self

    def predict(self, raw: np.ndarray) -> np.ndarray:
        return self._lr.predict_proba(raw.reshape(-1, 1))[:, 1]

    @property
    def coef(self) -> tuple[float, float]:
        return float(self._lr.coef_[0][0]), float(self._lr.intercept_[0])


class IsotonicCalibrator:
    def __init__(self):
        self._iso = IsotonicRegression(y_min=0.0, y_max=1.0,
                                       increasing=True, out_of_bounds="clip")

    def fit(self, raw: np.ndarray, y: np.ndarray):
        self._iso.fit(raw, y)
        return self

    def predict(self, raw: np.ndarray) -> np.ndarray:
        return self._iso.predict(raw)


def calib_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {"brier": brier(y, p), "log_loss": log_loss_(y, p), "ece": ece(y, p)}


def weekly_calibration(scores: pd.DataFrame, p_col: str, weeks: np.ndarray) -> list[dict]:
    rows = []
    for w in weeks:
        m = (scores["week"] == w).to_numpy()
        if m.sum() < 500:
            continue
        y = scores.loc[m, "y"].to_numpy()
        p = scores.loc[m, p_col].to_numpy()
        rows.append({"week": int(w), "n": int(m.sum()),
                     "brier": brier(y, p), "ece": ece(y, p),
                     "mean_p": float(p.mean()), "fraud_rate": float(y.mean())})
    return rows


def main() -> None:
    scores = pd.read_parquet(config.DATA_PROCESSED / "scores.parquet")
    selected = load_json("stage2_model_selection")["selected"]
    if selected == "lgbm":
        raw_all = scores["lgbm_raw"].to_numpy(dtype="float64")
        p_uncal_all = scores["lgbm_p"].to_numpy(dtype="float64")
    else:  # champion selected: use logit of its probability as the raw score
        p = np.clip(scores["lr_p"].to_numpy(dtype="float64"), EPS, 1 - EPS)
        raw_all = np.log(p / (1 - p))
        p_uncal_all = p

    val = (scores["month"] == config.VAL_MONTH).to_numpy()
    y_val = scores.loc[val, "y"].to_numpy()
    raw_val = raw_all[val]

    platt = PlattCalibrator().fit(raw_val, y_val)
    iso = IsotonicCalibrator().fit(raw_val, y_val)

    p_val = {"uncalibrated": p_uncal_all[val],
             "platt": platt.predict(raw_val),
             "isotonic": iso.predict(raw_val)}
    metrics = {name: calib_metrics(y_val, p) for name, p in p_val.items()}
    chosen = min(("platt", "isotonic"), key=lambda m: metrics[m]["brier"])

    # apply frozen calibrators to every row for downstream stages
    scores["p_platt"] = platt.predict(raw_all).astype("float32")
    scores["p_isotonic"] = iso.predict(raw_all).astype("float32")
    scores["p_uncal"] = p_uncal_all.astype("float32")
    scores["p_cal"] = scores[f"p_{chosen}"]
    scores.to_parquet(config.DATA_PROCESSED / "scores.parquet", index=False)

    val_weeks = np.unique(scores.loc[val, "week"])
    payload = {
        "selected_model": selected,
        "val_metrics": metrics,
        "chosen_method": chosen,
        "platt_coef": platt.coef,
        "reliability_val": {name: reliability_bins(y_val, p)
                            for name, p in p_val.items()},
        "weekly_val_calibration": {
            "uncalibrated": weekly_calibration(scores, "p_uncal", val_weeks),
            "calibrated": weekly_calibration(scores, "p_cal", val_weeks)},
        "note": "calibrators fit on month-4 validation only; test untouched",
    }
    save_json(payload, "stage3_calibration")
    log.info("calibration on val: %s", {k: round(v['brier'], 5)
                                        for k, v in metrics.items()})
    log.info("chosen method: %s", chosen)


if __name__ == "__main__":
    main()
