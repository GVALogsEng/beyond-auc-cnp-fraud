# Beyond AUC: A Cost-Sensitive, Calibration-First Evaluation of Card-Not-Present Fraud Models under Temporal Drift

**Geoff [LASTNAME]** — New York University
*Code and full experiment log: this repository. All numbers below are produced by the committed pipeline and persisted in `results/metrics/`.*

## Abstract

Card-not-present (CNP) fraud detection is usually reported as a ranking problem: a model, a shuffled cross-validation, an AUC. Production fraud systems answer a different question — *which transactions should be declined, at what dollar cost, next month* — and the usual evaluation answers it badly. Using the IEEE-CIS dataset (590,540 e-commerce transactions, 3.50% fraud), we build a deliberately standard model stack (regularized logistic regression, LightGBM) and spend the evaluation budget where production risk actually lives. Under a strict out-of-time protocol (train months 0–3, validate on month 4, touch month 5 once), random 5-fold cross-validation overstates test-month performance by **0.30 PR-AUC and 22 points of TPR@5%FPR** for the gradient-boosted challenger (0.87→0.56; 0.89→0.67) while barely moving the linear champion — the optimism tracks a model's capacity to memorize entities. A two-parameter cost model (missed fraud costs the amount; a false decline costs k × amount) admits an amount-independent optimal threshold p\* = k/(1+k) — but only on calibrated probabilities. On the test month the analytic threshold on calibrated scores beats the F1-optimal threshold by **+7.9 points of savings** (paired 95% CI +5.1 to +10.8) and cost-thresholding the raw class-weighted scores by +2.2 points ($11.1K on the month; paired CI −0.2 to +4.3). The final system attains **39.4% cost savings** against the best trivial policy at the central k = 0.15 (95% CI 35.8–42.8%), with TPR at the 5%-FPR operating point of **67.2%** (65.5–68.8%). Weekly decay curves, PSI, and adversarial validation quantify drift: monthly retraining recovers ~7.5 points of the savings the static model loses in month 5.

## 1. Introduction

The dominant public benchmark treats fraud detection as pure ranking, but the economics of CNP payments are decision-theoretic and sharply asymmetric. Javelin Strategy & Research estimated falsely declined card transactions at almost **$118 billion per year** in the U.S. — with 15% of cardholders experiencing a false decline within a year and 39% of those abandoning the declined card afterward [4]. The problem has not aged away: in PYMNTS Intelligence's 2026 merchant survey, **85% of merchants** call reducing friction for legitimate customers their biggest fraud-prevention challenge, and nearly half estimate that **up to 5% of legitimate orders** are incorrectly declined, roughly $50 billion of industry revenue [5]; the 2024 MRC/Visa Acceptance global fraud report finds most merchants self-report false-positive rates of 2–10% of orders [3]. Meanwhile the threat side keeps industrializing — Visa's Spring 2025 Biannual Threats Report tracks a 22% six-month rise in enumerated transactions and attributes ~US$1.1B of annual follow-on fraud to enumeration alone [7].

This tension — fraud loss vs. false-decline loss — is precisely the objective Visa assigns to its own CNP scoring product: Visa Deep Authorization is described as a deep-learning risk score built to "boost approvals for card-not-present transactions" while detecting and declining fraud [6]. A model that improves AUC but is thresholded naively, on miscalibrated scores, can *lose* money relative to a weaker model with an honest decision layer. That decision layer, not the classifier, is this project's subject.

We make four contributions on the public IEEE-CIS dataset [1]:

1. **A cost-sensitive decision layer evaluated in dollars.** With FN cost = amount and FP cost = k x amount, expected-cost decisioning declines iff p > k/(1+k) (Elkan [2]) — an amount-independent threshold we derive in one line, stress across k ∈ {0.05, …, 1.0}, and compare against F1-optimal, Youden-J, and trivial policies using Bahnsen-convention savings [10], with bootstrap confidence intervals, on a strictly out-of-time test month.
2. **A measured optimism gap.** The same models, features, and hyperparameters are evaluated under random 5-fold CV, GroupKFold by month, and the temporal protocol; the difference between the first and last is the optimism a leaderboard-style evaluation silently adds.
3. **Calibration as a load-bearing component, priced in dollars.** Class-weighted training deliberately distorts probabilities; we repair them with Platt/isotonic calibrators fit only on the validation month [12], and price the uncalibrated-vs-calibrated difference at the analytic threshold.
4. **Drift made operational.** Weekly out-of-time decay under three retraining cadences (static / expanding / sliding), PSI on the model's top features, and adversarial validation — the monitoring quantities a model-risk owner would actually track.

Everything is CPU-reproducible (`make all`), every aggregation feature is provably causal (property-tested), and the test month is touched exactly once.

## 2. Related work

**Cost-sensitive learning.** Elkan's foundational treatment shows that for a binary decision with per-example costs, the optimal decision on calibrated probabilities is a threshold determined by the cost matrix alone [2]. Bahnsen, Aouada & Ottersten develop example-dependent cost-sensitive methods for card fraud and introduce the *savings* convention used here — cost improvement over the best trivial (accept-all / decline-all) policy [10]. We deliberately keep the model conventional and put the cost-sensitivity in the decision layer, where it can be audited.

**Calibration.** Platt scaling fits a sigmoid on a held-out score sample [12a]; isotonic regression fits a monotone step function and needs more data but no shape assumption (Zadrozny & Elkan [12b]). Guo et al. document how modern learners are systematically miscalibrated and how simple post-hoc maps repair them [12c]. Fraud adds two twists: class-weighting (used here) shifts scores by design, and drift decays calibration over time — we track weekly Brier/ECE out-of-time for exactly that reason.

**IEEE-CIS top solutions.** The winning entry (Deotte & Yakovlev) reconstructed a client identifier from `card1`, `addr1` and `D1` and aggregated behavior over it, validating with GroupKFold using months as non-overlapping groups because client-level memorization otherwise masquerades as generalization [8]. We cite it as *precedent and warning*: the same mechanics that win a leaderboard are the ones that leak under a shuffled split. Our aggregates are restricted to expanding, past-only statistics, and no label-derived aggregate is used at all (production chargeback labels arrive weeks late).

**Drift and operating points.** The Bank Account Fraud benchmark (Jesus et al., NeurIPS 2022) builds dynamic, imbalanced tabular datasets and standardizes evaluation at a fixed operating point — "we select the threshold in order to obtain 5% false positive rate (FPR), and measure the true positive rate (TPR) at that point" [11]. We adopt TPR@5%FPR as the headline ranking metric and add the dollar-denominated layer on top.

## 3. Data

The IEEE-CIS Fraud Detection corpus (Vesta Corporation, 2019) contains **590,540** e-commerce transactions over **182 days**, of which **20,663 (3.499%)** are fraudulent; **24.4%** of transactions have device/identity enrichment in a second table joined on `TransactionID` [1]. Feature blocks: core transaction fields (amount, product code, card, address, distance, purchaser/recipient email domains), counts `C1–C14`, time deltas `D1–D15`, match flags `M1–M9`, 339 anonymized Vesta-engineered features `V1–V339`, and 41 identity columns. `TransactionDT` is a second-offset from an undisclosed reference; we define `day = TransactionDT // 86400`, `week = day // 7`, `month = min(day // 30, 5)` (the 2–3 day tail stub is merged into the final month; DECISIONS D-003).

Two facts shape everything downstream. First, the stream is **non-stationary**: monthly fraud rate moves from 2.48% (month 0) through 4.04% (month 1) down to 3.47% (month 5), and transaction volume swings weekly (Figure 1, Figure 2). Second, the **label is operational, not oracular**: `isFraud` reflects Vesta's chargeback-linked labeling; downstream transactions of a flagged account/card can inherit the label, and reporting delay censors the newest data. Both properties argue for chronological evaluation and against label-based feature aggregation. All modeling below uses the labeled training file only; Kaggle's unlabeled test set appears nowhere in any evaluation claim.

![Figure 1 — class balance and amount distributions](reports/figures/figure_01_class_balance_amounts.png)

![Figure 2 — weekly volume and fraud rate, with the temporal split](reports/figures/figure_02_volume_fraud_over_time.png)

## 4. Methodology

### 4.1 Temporal protocol

![Figure P — the fixed temporal protocol](reports/figures/figure_00_protocol.png)

Train: months 0–3 (410,601 rows). Validation: month 4 (85,303 rows) — used for calibration fitting, threshold selection, and model selection. Test: month 5 (94,636 rows) — evaluated exactly once, after every upstream choice is frozen. Hyperparameters are selected with GroupKFold using month as the group within the training window. Weekly curves use `week = day // 7`. Automated tests assert split integrity (no temporal overlap; fold groups disjoint) and feature causality (see 4.3).

### 4.2 Models and imbalance

The **champion** is L2-regularized logistic regression on interpretable blocks (core, counts, time deltas, match flags, numeric identity), with window-fit medians/scaling, missingness indicators for the D block, and top-30 one-hot vocabularies fit on the training window only. The **challenger** is LightGBM on all blocks with native categorical handling. Imbalance is handled with `scale_pos_weight` (= neg/pos of each fit window); we do not use SMOTE or any synthetic oversampling, because resampling distorts the probability scale that the decision layer depends on and interacts badly with post-hoc calibration — weighting plus explicit recalibration achieves the same operating-point control while keeping the probability semantics auditable (DECISIONS D-004/D-005). Search is deliberately modest (Optuna TPE, 12 trials inside a 45-minute budget, GroupKFold-by-month folds; best fold PR-AUC 0.626); rigor is spent on evaluation, not tuning depth.

### 4.3 Causal features

Beyond row-local transforms (log-amount, cents pattern, hour, day-of-week), we add expanding **past-only** aggregates per `card1` and per `(card1, addr1)`: prior-transaction count, seconds since the entity's previous transaction, past mean amount, and current/past-mean amount ratio. `tests/test_causal_features.py` enforces the defining property by *prefix recomputation*: the feature value at row i is identical when every later row is deleted. No aggregate uses `isFraud` — chargeback labels arrive with multi-week delay in production, and past-label features would silently assume instant label availability.

### 4.4 Cost framework

Let a transaction have amount A and fraud probability p. Approving a fraud costs A (write-off proxy); declining a legitimate order costs kA (lost margin plus attrition proxy). Expected costs: approve → pA; decline → (1−p)kA. Decline exactly when

**pA > (1−p)kA  ⟺  p > k/(1+k)**,

and A cancels: the optimal threshold is *amount-independent* — but the inequality is a statement about a **probability**, so it is only actionable on calibrated scores. This is why calibration is load-bearing rather than cosmetic. Central case k = 0.15 (threshold 0.1304); sensitivity grid k ∈ {0.05, 0.15, 0.30, 0.60, 1.0}. Savings follow Bahnsen et al. [10]: `savings = 1 − Cost_model / Cost_baseline` with `Cost_baseline = min(cost(approve-all), cost(decline-all))`. Six policies are compared on identical test data: approve-all, decline-all, F1-optimal and Youden-J thresholds (selected on validation), and the analytic cost threshold applied to uncalibrated vs. calibrated probabilities. Platt and isotonic calibrators are fit on month 4 only; the method with lower validation Brier is frozen for test.

### 4.5 Drift and ablation design

Weekly decay is evaluated over months 4–5 for three cadences — **static** (train once on months 0–3), **expanding** (retrain monthly on all past), **sliding** (trailing 3 months) — under a uniform rule: every window holds out its final 14 days for calibration (D-006). Input drift is tracked with PSI (train window → each evaluation week) for the model's top-20 gain features, plus adversarial validation (a classifier separating train rows from out-of-time rows; AUC ≈ 0.5 means indistinguishable), run both with and without the cumulative aggregate features. Incremental validity (Stage 6) retrains the challenger under fixed hyperparameters and tree count on forward-addition (A → A+…+F) and leave-one-block-out configurations, scoring ΔTPR@5%FPR and Δsavings on validation; the selected configuration is confirmed once on test.

## 5. Results

*This section reports the persisted outputs in `results/metrics/`; every figure is generated by `make figures` from those artifacts.*

### 5.1 The optimism gap

Identical features and tuned hyperparameters, three split regimes:

| LightGBM | ROC-AUC | PR-AUC | TPR@5%FPR |
|---|---|---|---|
| random 5-fold CV (mean) | 0.968 | 0.866 | 0.891 |
| GroupKFold by month (mean) | 0.917 | 0.618 | 0.700 |
| out-of-time test (month 5) | 0.904 | 0.564 | 0.672 |
| **gap: random − test** | **+0.064** | **+0.302** | **+0.219** |

The logistic champion's corresponding gaps are +0.020 / +0.017 / +0.035. Two observations matter. First, the shuffled estimate is not mildly optimistic — it reports a model roughly *30 PR-AUC points* better than the one that exists next month, and the distortion is largest exactly where fraud metrics live (the high-precision region; ROC-AUC hides it). Second, the gap is model-dependent: the boosted model, which can memorize card-profile behavior across a shuffled split, inflates; the linear model, which cannot, does not. Grouped-by-month CV lands within 3–5 points of the true out-of-time numbers and is the honest offline surrogate. **Takeaway (Figure 3): the choice of split changes the reported number more than any modeling choice in this paper.**

![Figure 3 — random-split vs honest evaluation](reports/figures/figure_03_optimism_gap.png)

### 5.2 Calibration

Class-weighted training (`scale_pos_weight` ≈ 27) leaves the raw scores usable for ranking but distorted as probabilities (Platt slope 0.67 on validation: raw log-odds are overconfident). Isotonic regression, fit on month 4 only, wins on validation Brier (0.01802 vs 0.01820 Platt vs 0.01866 raw) and transfers to the untouched test month: Brier 0.0217 → 0.0212, log-loss 0.1053 → 0.0884, ECE 0.0146 → 0.0041. Weekly out-of-time tracking shows the calibrated scores hold their reliability across both evaluation months. **Takeaway (Figure 4): a one-month held-out calibrator repairs the probability scale that the decision layer is about to lean on — cheaply and stably.**

![Figure 4 — reliability before/after calibration](reports/figures/figure_04_reliability.png)

### 5.3 The decision layer in dollars

Test month: 94,636 transactions, 3.47% fraud; baseline policy is approve-all (cost $495,244). All thresholds frozen on validation.

| policy | total cost | savings | fraud $ caught | legit $ declined | TPR | FPR |
|---|---|---|---|---|---|---|
| approve all | $495,244 | 0.0% | $0 | $0 | 0.000 | 0.000 |
| decline all | $1,872,497 | −278.1% | $495,244 | $12,483,317 | 1.000 | 1.000 |
| F1-optimal | $339,675 | 31.4% | $171,157 | $103,921 | 0.439 | 0.007 |
| Youden J | $359,583 | 27.4% | $367,605 | $1,546,285 | 0.748 | 0.089 |
| cost thr., uncalibrated | $311,403 | 37.1% | $221,430 | $250,591 | 0.521 | 0.015 |
| **cost thr., calibrated** | **$300,271** | **39.4%** [35.8, 42.8] | $274,003 | $526,866 | 0.604 | 0.029 |

The ordering is the argument. Youden J catches the most fraud dollars but burns $1.55M of legitimate volume to do it; F1 is precise but timid; the analytic threshold k/(1+k) beats both — **+7.9 points of savings over F1** (paired 95% CI [+5.1, +10.8], positive in 100% of resamples). Calibrating before thresholding adds +2.2 points / $11.1K on the month (2.25% of baseline); the paired CI [−0.2, +4.3] just crosses zero, so on a single test month this increment is directionally consistent (96% of resamples) rather than individually significant — we report it as such. Across the k grid (Figure 6), the analytic policy adapts and stays positive everywhere (58.6% savings at k = 0.05, 19.3% at k = 1.0), while the fixed Youden threshold collapses to **−238%** at k = 1.0: a threshold chosen without reference to costs does not merely underperform, it changes sign. **Takeaway (Figure 5): where you cut matters more than how well you rank — and the right cut is a formula, not a grid search, once probabilities are calibrated.**

![Figure 5 — savings vs threshold](reports/figures/figure_05_cost_vs_threshold.png)

![Figure 6 — sensitivity across k](reports/figures/figure_06_sensitivity_heatmap.png)

### 5.4 Drift and retraining cadence

Weekly performance decays visibly within the two out-of-time months, with a trough at week 20 and only partial recovery. Retraining at the month boundary changes the picture (month-5 means, uniform holdout-calibration protocol):

| cadence | TPR@5%FPR | PR-AUC | savings (k=0.15) |
|---|---|---|---|
| static (train once) | 0.651 | 0.535 | 0.354 |
| expanding (all past) | **0.703** | **0.600** | **0.429** |
| sliding (last 3 months) | 0.696 | 0.597 | 0.422 |

One monthly retrain recovers **+7.5 points of savings** over the static model; keeping all history edges out the 3-month window, so recency matters less than freshness. On the input side, PSI flags concentrated, interpretable drift: `id_31` (browser version) reaches PSI 1.36 — software-ecosystem churn, invisible to a static model — and the cumulative entity counters drift mechanically (0.37); 3 of the top-20 features breach the conventional 0.2 alarm by the final week. Adversarial validation separates train from out-of-time rows with AUC 0.964 — and still 0.899 after removing the mechanically-trending aggregate features, so the shift is genuine covariate drift, not an artifact of the counters. **Takeaway (Figures 7–8): this dataset retires "train once, deploy forever" on its own — and the decision-relevant decay (savings) is steeper than the rank-metric decay.**

![Figure 7 — weekly decay by retraining policy](reports/figures/figure_07_decay_curves.png)

![Figure 8 — PSI and adversarial validation](reports/figures/figure_08_psi_adversarial.png)

### 5.5 Incremental validity of the feature blocks

Forward addition on validation (fixed hyperparameters and tree count):

| configuration | TPR@5%FPR | savings |
|---|---|---|
| A core (+ causal aggregates) | 0.439 | 0.262 |
| + B counts (C1–C14) | 0.616 | 0.405 |
| + C time deltas (D1–D15) | 0.656 | 0.432 |
| + D match flags (M1–M9) | 0.666 | 0.449 |
| + E Vesta (V1–V339) | 0.676 | 0.473 |
| + F identity | 0.685 | 0.470 |

Core + counts — 40 interpretable features — deliver **86% of the full configuration's savings**. The 339 anonymized Vesta features buy +2.4 points of savings; the identity table buys +0.9 points of TPR while costing −0.3 points of savings at this operating point. Leave-one-block-out confirms no block strictly harms both criteria (removing counts costs −4.9 points of savings; removing match flags is cost-neutral), so the full configuration is retained for the headline model. **Takeaway (Figure 9): most of the black-box's dollar value lives in a small auditable core — a governance-relevant fact when every extra block is attack surface, latency, and model-risk documentation.**

![Figure 9 — forward addition and leave-one-block-out](reports/figures/figure_09_ablation.png)

### 5.6 Explainability

TreeSHAP on a stratified 20K-row test sample puts velocity and identity-consistency signals at the top: `C13` and `C1` (Vesta count features), purchaser email domain, and — notably — three of the causal aggregates built here (`card1_past_amt_mean`, `card1_addr1_past_count`, `card1_past_count`) rank inside the top eleven. The paired waterfalls show the decision anatomy a dispute analyst needs: the true positive is declined on a first-seen card profile transacting at an unusual amount; the false positive is a legitimate customer who *looks* first-seen — exactly the failure mode that step-up authentication, rather than a hard decline, should absorb. **Takeaway (Figure 10): the model's dollars come from behavioral-consistency features, which is also where its false declines come from.**

![Figure 10a — TreeSHAP beeswarm](reports/figures/figure_10a_shap_beeswarm.png)

![Figure 10b — true positive waterfall](reports/figures/figure_10b_waterfall_tp.png)
![Figure 10c — false positive waterfall](reports/figures/figure_10c_waterfall_fp.png)

## 6. Discussion

**Limitations.** This is one anonymized 2019 e-commerce dataset; the drift studied is historical, not adversarially live. The label is chargeback-linked and inherits reporting noise, delay, and within-account propagation — our "ground truth" is itself an operational artifact. The cost model is a two-parameter proxy: real FN costs include interchange, disputes, and recovery; real FP costs are heterogeneous across customers and partially recoverable through step-up authentication rather than hard declines; k is best read as a governance dial, which is why we report the full sensitivity grid rather than defend a single value. Calibrators and thresholds share the validation month; with more months one would separate those roles.

**What changes at Visa scale.** An issuer-side CNP score runs at authorization time under a roughly millisecond latency budget, which constrains feature stores to streaming, incrementally-updatable state (our expanding aggregates are the batch shadow of that design). Decisions feed back into the data: declined transactions never earn labels, creating selective label censoring that offline benchmarks cannot exhibit. Adversaries adapt to the deployed boundary — enumeration attacks alone shifted 22% in six months [7] — so the drift measured here is a lower bound on the operational problem. And at network scale, calibration is not a nicety but the contract: issuers consume scores as probabilities when setting strategy thresholds.

**A monitoring plan this analysis directly supports.** (1) Weekly TPR@5%FPR and savings on a rolling labeled window, alarmed against the decay slopes in Figure 7. (2) Weekly Brier/ECE on the deployed calibrator; recalibrate when the uncalibrated-vs-calibrated dollar gap reopens. (3) PSI on the top-20 features with the conventional 0.2 trigger, plus a quarterly adversarial-validation AUC as an omnibus shift detector. (4) Retraining cadence set by the measured curves — retrain when its projected savings recovery exceeds retraining and review cost, not on a calendar reflex. (5) Champion–challenger with the logistic champion as a sanity floor and threshold changes governed as decisions, not tuning.

## 7. Reproducibility

Python 3.11; pinned versions in `requirements.txt`; seed 42 throughout. Data (Kaggle credentials + accepted competition rules required):

```bash
kaggle competitions download -c ieee-fraud-detection -p data/raw/
# or place ieee-fraud-detection.zip at data/raw/ manually
make data      # checksum, unpack, schema validation report
make features  # causal feature build + EDA artifacts
make train     # tuning + three split regimes + final model + scores
make evaluate  # calibration, cost layer, ablation, drift, single test pass
make figures   # all paper figures from persisted artifacts
make test      # split-integrity, causality, and cost-model tests
make notebooks # generate + execute the six thin notebooks
make app       # streamlit demo from cached artifacts (no Kaggle needed)
```

Raw data is never committed. SHA-256 checksums of the archive and CSVs are recorded in `results/metrics/data_checksums.json`. Wall-clock on the 2-vCPU build machine: `make data`+`features` ≈ 6 min; `make train` ≈ 2 h 03 m (45 m of it the capped tuning budget); `make evaluate` ≈ 2 h 33 m (ablation 55 m, drift 31 m, final test pass 67 m — of which TreeSHAP on 20K rows × 2,190 trees is 65 m); figures + notebooks ≈ 3 m. Total ≈ **4 h 45 m**; on a typical 4–8-core laptop LightGBM's near-linear thread scaling brings this into the 2–3 h range. The Streamlit demo runs entirely from `app/artifacts/` (a cached, stratified 20K-row test sample); deployment notes for Streamlit Community Cloud / Hugging Face Spaces are in `app/README-deploy.md`, including the competition-rules caveat on redistributing data-derived artifacts.

## References

1. IEEE-CIS Fraud Detection (Kaggle competition, Vesta Corporation, 2019). https://www.kaggle.com/competitions/ieee-fraud-detection
2. Elkan, C. (2001). The Foundations of Cost-Sensitive Learning. *IJCAI-01*. https://cseweb.ucsd.edu/~elkan/rescale.pdf
3. Merchant Risk Council, Visa Acceptance Solutions & Verifi (2024). *2024 Global eCommerce Payments & Fraud Report* (25th ed.). https://www.cybersource.com/content/dam/documents/campaign/fraud-report/global-fraud-report-2024.pdf
4. Javelin Strategy & Research (2015). *False-Positive Card Declines Push Consumers to Abandon Issuers and Merchants.* https://javelinstrategy.com/press-release/false-positive-card-declines-push-consumers-abandon-issuers-and-merchants
5. PYMNTS Intelligence (2026). *Orchestrating Trust: The Future of Fraud Prevention in Payments* (merchant survey coverage). https://www.pymnts.com/fraud-prevention/2026/47-percent-of-merchants-report-false-declines-cost-them-sales/ ; https://www.pymnts.com/fraud-prevention/2026/85percent-of-merchants-say-fraud-tools-must-reduce-checkout-friction/
6. Visa (2024). Visa Deep Authorization — Visa Protect. https://www.visa.com/en-us/solutions/secure-card-payments
7. Visa Payment Ecosystem Risk & Control (2025). *Biannual Threats Report, Spring 2025.* https://corporate.visa.com/content/dam/VCOM/corporate/solutions/documents/visa-perc-biannual-report-spring-2025.pdf
8. Deotte, C. & Yakovlev, K. (2019). IEEE-CIS 1st-place solution write-up (FraudSquad). https://www.kaggle.com/competitions/ieee-fraud-detection/writeups/fraudsquad-1st-place-solution-part-2 ; accessible summary: https://developer.nvidia.com/blog/leveraging-machine-learning-to-detect-fraud-tips-to-developing-a-winning-kaggle-solution/
9. *(Anchor dropped after verification — see results/metrics/citation_verification.json.)*
10. Correa Bahnsen, A., Aouada, D. & Ottersten, B. (2015). Example-dependent cost-sensitive decision trees. *Expert Systems with Applications* 42(8), 6609–6619. https://albahnsen.github.io/files/Example-Dependent%20Cost-Sensitive%20Decision%20Trees.pdf
11. Jesus, S., Pombal, J., Alves, D., Cruz, A., Saleiro, P., Ribeiro, R.P., Gama, J. & Bizarro, P. (2022). Turning the Tables: Biased, Imbalanced, Dynamic Tabular Datasets for ML Evaluation. *NeurIPS Datasets & Benchmarks*. https://papers.nips.cc/paper_files/paper/2022/hash/d9696563856bd350e4e7ac5e5812f23c-Abstract-Datasets_and_Benchmarks.html
12. (a) Platt, J. (1999). Probabilistic Outputs for Support Vector Machines… *Advances in Large Margin Classifiers* 10(3), 61–74. https://www.semanticscholar.org/paper/42e5ed832d4310ce4378c44d05570439df28a393 — (b) Zadrozny, B. & Elkan, C. (2002). Transforming classifier scores into accurate multiclass probability estimates. *KDD '02*. https://dl.acm.org/doi/10.1145/775047.775151 — (c) Guo, C., Pleiss, G., Sun, Y. & Weinberger, K.Q. (2017). On Calibration of Modern Neural Networks. *ICML*. https://arxiv.org/abs/1706.04599

*Label caveat, dataset licensing, and the full decision log: `DECISIONS.md`. The dataset is subject to the Kaggle competition rules and is not redistributed by this repository.*
