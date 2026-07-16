"""LightGBM challenger: parameter space, modest Optuna tuning, fit helpers.

Tuning protocol (spec): GroupKFold with month as the group, within the
training months only. To respect the CPU budget the search runs on a
stratified row subsample and is capped by both trial count and wall-clock
(config.TUNING_*); the selected parameters are then refit on full windows.
"""
from __future__ import annotations

import time

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold

from src import config
from src.utils import get_logger, save_json

log = get_logger("models.lgbm")

BASE_PARAMS = dict(
    objective="binary",
    metric="average_precision",
    verbosity=-1,
    n_jobs=2,
    random_state=config.SEED,
    max_bin=255,
)


def make_model(params: dict, scale_pos_weight: float, n_estimators: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(**BASE_PARAMS, **params,
                              scale_pos_weight=scale_pos_weight,
                              n_estimators=n_estimators)


def pos_weight(y: np.ndarray) -> float:
    pos = float(y.sum())
    return (len(y) - pos) / max(pos, 1.0)


def fit_with_early_stop(params: dict, X_tr: pd.DataFrame, y_tr: np.ndarray,
                        X_va: pd.DataFrame, y_va: np.ndarray):
    model = make_model(params, pos_weight(y_tr), config.MAX_ESTIMATORS)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
              eval_metric="average_precision",
              callbacks=[lgb.early_stopping(config.EARLY_STOPPING_ROUNDS, verbose=False),
                         lgb.log_evaluation(0)])
    return model


def _sample_params(trial: optuna.Trial) -> dict:
    return dict(
        num_leaves=trial.suggest_int("num_leaves", 31, 255, log=True),
        learning_rate=trial.suggest_float("learning_rate", 0.02, 0.08, log=True),
        min_child_samples=trial.suggest_int("min_child_samples", 20, 300, log=True),
        feature_fraction=trial.suggest_float("feature_fraction", 0.5, 0.9),
        bagging_fraction=trial.suggest_float("bagging_fraction", 0.6, 1.0),
        bagging_freq=1,
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
    )


DEFAULT_PARAMS = dict(num_leaves=96, learning_rate=0.05, min_child_samples=80,
                      feature_fraction=0.7, bagging_fraction=0.9, bagging_freq=1,
                      reg_alpha=0.1, reg_lambda=1.0)


def tune(X: pd.DataFrame, y: np.ndarray, months: np.ndarray) -> dict:
    """Optuna TPE over GroupKFold(month) on a stratified row subsample."""
    rng = np.random.RandomState(config.SEED)
    idx = np.arange(len(y))
    keep = np.zeros(len(y), dtype=bool)
    for cls in (0, 1):
        cls_idx = idx[y == cls]
        take = rng.choice(cls_idx, size=int(len(cls_idx) * config.TUNING_ROW_FRACTION),
                          replace=False)
        keep[take] = True
    Xs, ys, ms = X[keep], y[keep], months[keep]
    log.info("tuning subsample: %d rows (%.1f%% fraud)", keep.sum(), 100 * ys.mean())

    gkf = GroupKFold(n_splits=len(np.unique(ms)))
    # GroupKFold by month; to respect the CPU budget the search scores each
    # trial on the two folds whose held-out months are most recent (the
    # mechanism the spec fixes, evaluated on fewer splits -- DECISIONS D-010).
    fold_months = []
    all_folds = []
    for tr_i, va_i in gkf.split(Xs, ys, groups=ms):
        all_folds.append((tr_i, va_i))
        fold_months.append(int(np.max(ms[va_i])))
    order = np.argsort(fold_months)[::-1]
    folds = [all_folds[i] for i in order[:2]]

    def objective(trial: optuna.Trial) -> float:
        params = _sample_params(trial)
        scores, iters = [], []
        for tr_i, va_i in folds:
            model = fit_with_early_stop(params, Xs.iloc[tr_i], ys[tr_i],
                                        Xs.iloc[va_i], ys[va_i])
            p = model.predict_proba(Xs.iloc[va_i])[:, 1]
            scores.append(average_precision_score(ys[va_i], p))
            iters.append(model.best_iteration_ or config.MAX_ESTIMATORS)
        trial.set_user_attr("best_iters", iters)
        return float(np.mean(scores))

    sampler = optuna.samplers.TPESampler(seed=config.SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    t0, n_done = time.time(), 0
    while n_done < config.TUNING_MAX_TRIALS:
        study.optimize(objective, n_trials=1)
        n_done += 1
        elapsed = time.time() - t0
        per_trial = elapsed / n_done
        log.info("trial %d/%d  best=%.5f  (%.0fs/trial)", n_done,
                 config.TUNING_MAX_TRIALS, study.best_value, per_trial)
        if n_done >= 4 and elapsed + per_trial > config.TUNING_TIME_BUDGET_S:
            log.info("stopping search at time budget (%.0fs elapsed)", elapsed)
            break

    best = dict(study.best_trial.params)
    best["bagging_freq"] = 1
    payload = {"best_params": best,
               "best_cv_pr_auc": study.best_value,
               "n_trials": n_done,
               "elapsed_s": time.time() - t0,
               "best_iters_per_fold": study.best_trial.user_attrs.get("best_iters"),
               "subsample_fraction": config.TUNING_ROW_FRACTION,
               "protocol": "Optuna TPE, GroupKFold by month within train months",
               "trials": [{"number": t.number, "value": t.value, "params": t.params}
                          for t in study.trials]}
    save_json(payload, "stage2_lgbm_tuning")
    return best
