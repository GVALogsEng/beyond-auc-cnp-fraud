# Run report for Geoff — Beyond AUC (IEEE-CIS CNP fraud)

**Build finished:** 2026-07-16 ~05:30 UTC · **Total pipeline wall-clock:** ≈ 4 h 45 m on the 2-vCPU cloud container · **Tests:** 18/18 green · **README audit:** all 71 numeric claims machine-checked against `results/metrics/` — zero mismatches.

## Stages completed

All ten: Stage 0 gate (SHA-256-verified archive) → EDA + leakage audit → baselines + optimism gap → calibration → cost layer → drift/retraining → ablation → SHAP → README pre-print + Streamlit app → template narratives. Six notebooks generated and executed; `streamlit run app/app.py` boots clean from cached artifacts.

## Headline numbers (test month, touched once)

- **Savings at k = 0.15: 39.4%** of achievable cost vs approve-all baseline, bootstrap 95% CI **[35.8%, 42.8%]** — total cost $300,271 vs $495,244 baseline on 94,636 transactions.
- **TPR @ 5% FPR: 67.2%** [65.5%, 68.8%]; PR-AUC 0.564 [0.547, 0.581]; ROC-AUC 0.904 (reported, deprioritized).
- **Optimism gap (the headline finding):** random 5-fold CV says PR-AUC 0.866 / TPR@5 0.891; the strict out-of-time test says 0.564 / 0.672. **Random CV overstates by 0.30 PR-AUC and 22 TPR points** — and the gap is model-dependent (logistic champion: ≤0.035 across all metrics). GroupKFold-by-month lands within 3–5 points of truth.
- **Calibration effect in dollars:** isotonic (chosen on val Brier) improves test ECE 0.0146 → 0.0041 and is worth **+2.2 savings points = $11,132 on the month** vs cost-thresholding raw scores — honest caveat: the paired CI [−0.2, +4.3] just crosses zero (positive in 96% of resamples). The *decisively* significant win is the analytic threshold vs F1-optimal: **+7.9 points, CI [+5.1, +10.8]**, and vs Youden the fixed threshold actually flips to −238% savings at k = 1.0.
- **Best retraining cadence:** monthly **expanding** — month-5 savings 42.9% vs 42.2% sliding vs 35.4% static (+7.5 points over never retraining). Freshness > recency.
- **Ablation verdict:** core + C-counts (40 features) = **86% of full-config savings**; V1–V339 add +2.4 points; identity adds TPR but −0.3 points of savings. Nothing dropped (no LOBO config improved both criteria).
- **Drift diagnostics:** adversarial validation AUC 0.964 (0.899 even excluding the mechanically-trending aggregates); top PSI: id_31 browser-version churn at 1.36; 3/20 features breach 0.2 by the final week.

## Deviations from spec (all logged in DECISIONS.md)

1. **D-001/D-009 — data acquisition:** kaggle.com and PyPI unreachable from the sandbox; you supplied the official zip + wheels through the folder bridge (large files split at 60 MB; reassembled archive SHA-256-matched your Mac's checksum exactly).
2. **D-008 — XGBoost omitted** (132 MB wheel wouldn't cross the bridge; spec names LightGBM primary).
3. **D-010 — tuning ran 12 Optuna trials** on the 2 most-recent GroupKFold month folds inside the 45-min budget (spec allowed ≤ ~30; 2 vCPUs made full-fold search infeasible).
4. **Citations:** 2 of the 12 spec anchors failed verification and were dropped/replaced (Datos/Cybersource 1.51%/$175B not in the actual report — replaced with its verified 2–10% false-positive figure; the 2026 arXiv OOT paper unrecoverable). Record: `results/metrics/citation_verification.json`.
5. **Stage 9 ran the template fallback** (no `ANTHROPIC_API_KEY` in the environment); 400 narratives cached; the API code path ships unexercised.
6. **Runtime 4 h 45 m** vs the 2–3 h target — 2 vCPUs; §7 documents per-stage times (TreeSHAP on the 2,190-tree model alone took 65 m).

## Open questions for you

- The final model is large (2,190 trees, num_leaves 170; 41 MB). Fine for GitHub, but if you want a leaner artifact I can add a compactness constraint and re-run tuning.
- Do you want the demo's cached sample replaced with synthetic rows before any public deploy? (See `app/README-deploy.md` — competition rules restrict redistribution; deploy privately or swap the sample.)
- Month 4 currently serves calibration + threshold + model selection; a 7-month dataset would let those roles separate. Worth a "limitations" bullet in interviews.

## Your to-do list

1. Replace `[LASTNAME]` in README.md and LICENSE; finalize the title if you want a different one.
2. `git remote add origin … && git push` (history is clean; data and app artifacts are gitignored).
3. Decide the demo deployment (local / private Space / synthetic sample) and follow `app/README-deploy.md`.
4. Resume bullet from real numbers, e.g.: *"Built a cost-sensitive, calibrated fraud-decision system on 590K CNP transactions with strict out-of-time evaluation; analytic cost thresholding beat F1-thresholding by 7.9 points of dollar savings (95% CI [5.1, 10.8]); showed random CV overstates PR-AUC by 0.30 and quantified the +7.5-point savings recovery from monthly retraining."*
5. Skim `DECISIONS.md` D-001…D-011 — it doubles as your interview story about working around constraints without compromising the evaluation.
