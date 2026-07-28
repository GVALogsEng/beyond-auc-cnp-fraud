# Beyond AUC: A Cost-Sensitive, Calibration-First Evaluation of Card-Not-Present Fraud Models under Temporal Drift

*Code and full experiment container. All numbers below are produced by the committed pipeline and persisted in `results/metrics/`.*

## Abstract
 
Card-not-present fraud is usually detected by models and then graded on how well they rank transactions from most to least suspicious. The production system ( live service that answers each payment request real time) must instead decide to approve or decline and its two errors carry opposite, unequal costs: a missed fraud costs the transaction amount; a false decline costs the margin on a legitimate sale. On the IEEE-CIS dataset, 590,540 real e-commerce transactions, 3.50% fraudulent, provided by the payment company Vesta for a Kaggle competition (2019), we keep the models conventional (regularized logistic regression; LightGBM, a gradient-boosted tree ensemble) and spend the evaluation effort on the decision layer that turns scores into actions, under a strict out-of-time protocol: train on the first four months, make every choice on the fifth, test exactly once on the sixth. Shuffled cross-validation, which mixes past and future rows, overstates next-month PR-AUC (ranking score suited to rare-event data) by 0.30 for LightGBM (0.87 → 0.56) while moving the linear model by 0.017: the optimism measures how much a model memorizes individual cards. With missed-fraud cost A and false-decline cost kA, expected cost is minimized by declining when the fraud probability exceeds k/(1+k), independent of A; on the untouched test month this threshold beats the conventional F1-selected one by +7.9 points of cost savings (paired 95% CI +5.1 to +10.8). The rule acts on probabilities, so the scores must first be calibrated: an isotonic calibrator — a monotone remapping of scores to observed fraud frequencies, fitted on one held-out month — adds +2.2 points (CI −0.2 to +4.3). The system saves 39.4% of the cost of the best model-free policy, approve-everything or decline-everything (CI 35.8–42.8%), catching 67.2% of fraud at a 5% false-positive rate; monthly retraining recovers +7.5 of the savings points a never-retrained model loses.
 
## 1. Introduction
 
In card-not-present (CNP) payments, within e-commerce and remote orders where the physical card is never shown, a fraud model reads each transaction as it arrives and returns a score: the higher, the more fraud-like. The customary evaluation treats that model as a *ranking* device. Data is split by shuffled cross-validation: rows are assigned to K-folds at random, the model trains on K−1 folds and is scored on the one held out, and the folds rotate until every row has been predicted once; an efficient default whose core assumption is that rows are interchangeable. The scores are then summarized by AUC, the area under the ROC curve, which is the probability that a randomly chosen fraud is scored above a randomly chosen legitimate transaction. AUC's appeal in this is that it is threshold-free which averages performance over every possible cutoff, so it can be reported without committing to one.
 
A payment system cannot avoid committing. It has to approve or decline each transaction, and the two errors this creates cost money in opposite directions. A missed fraud is approved, completes, and comes back as a chargeback: the transaction amount is lost. A false decline blocks a legitimate order: the margin on the sale is lost, and with some probability the customer is too. An evaluation that averages over all cutoffs is therefore silent on the two questions that matter operationally — where to place the cutoff, and what a misplaced one costs.
 
Neither error is insignificant. Javelin Strategy & Research estimated falsely declined card transactions at almost **$118 billion per year** in the U.S., with 15% of cardholders experiencing a false decline within a year and 39% of those abandoning the declined card afterward [4]. The problem has not dissipated: in PYMNTS Intelligence's 2026 merchant survey, **85% of merchants** call reducing friction for legitimate customers their biggest fraud-prevention challenge, and nearly half estimate that up to 5% of legitimate orders are incorrectly declined; roughly $50 billion of industry revenue [5]; the 2024 MRC/Visa Acceptance global fraud report finds most merchants self-report false-positive rates of 2–10% of orders [3]. The threat side, meanwhile, keeps industrializing: Visa's Spring 2025 Biannual Threats Report tracks a 22% six-month rise in enumerated transactions and attributes ~US$1.1B of annual follow-on fraud to enumeration alone [7].
 
Balancing these two losses is precisely the objective Visa assigns to its own CNP scoring product: Visa Deep Authorization is described as a deep-learning risk score built to "boost approvals for card-not-present transactions" while detecting and declining fraud [6]. The balance is struck not by the classifier but by the *decision layer*: the threshold that turns a score into approve-or-decline, and the probability scale that threshold is applied to. A model that improves AUC but is thresholded naively, on distorted scores, can lose money relative to a weaker model with an honest decision layer. This decision layer, not the classifier, is the project subject.
 
We make four contributions on the public IEEE-CIS dataset [1]:
 
1. **A cost-sensitive decision layer, evaluated in dollars.** With the cost of a missed fraud set to the transaction amount and the cost of a false decline set to k × amount, expected-cost decisioning declines exactly when the fraud probability exceeds k/(1+k) (Elkan [2]) — a threshold derived in one line (§4.4), independent of amount. We stress it across k ∈ {0.05, …, 1.0} and compare it against F1-optimal, Youden-J, and the trivial policies using the savings convention of Bahnsen et al. [10], with bootstrap confidence intervals, on a strictly out-of-time test month.
2. **A measured optimism gap.** The same models, features, and hyperparameters are evaluated three ways — shuffled 5-fold cross-validation, GroupKFold by month, and the temporal protocol. The difference between the first and the last is the optimism a leaderboard-style evaluation silently adds (§5.1).
3. **Calibration priced in dollars.** Training with class weights distorts the probability scale on purpose; we repair it with Platt/isotonic calibrators fitted only on the validation month [12] and price the repair at the analytic threshold (§5.2–5.3).
4. **Drift made operational.** Weekly out-of-time decay under three retraining cadences (static / expanding / sliding), population-stability indices on the model's top features, and adversarial validation — the quantities a model-risk owner would actually monitor (§5.4).
Everything runs on CPU from the public dataset (`make all`; obtaining the data requires a Kaggle account and acceptance of the competition rules). Every aggregation feature is tested to use only information from strictly earlier transactions — its value at any row is unchanged when all later rows are deleted (§4.3) — and the test month is touched exactly once.

## 2. Data
 
The IEEE-CIS Fraud Detection corpus (Vesta Corporation, 2019) contains **590,540** e-commerce transactions spanning **182 days**, of which **20,663 (3.499%)** are labeled fraudulent [1]. Each row carries the transaction amount, a product code, card and address fields, a distance field, and purchaser/recipient email domains, plus four blocks of features supplied with the data: counting features `C1–C14` (frequency counters whose exact definitions are withheld; the documented example is how many addresses are associated with the card); time deltas `D1–D15` (such as days since a previous transaction); binary match flags `M1–M9` (whether details on the card agree with details on the order); and `V1–V339`, features Vesta engineered in-house — rankings, counts, and other entity relations, anonymized. A second table, joined on `TransactionID`, adds 41 columns of device and network identity (device type and model, operating system, browser, and anonymized `id_` fields) for the **24.4%** of transactions that have it.
 
Time is provided as `TransactionDT`, a second-offset from an undisclosed reference date — the data reveal order and spacing, never the calendar. We define `day = TransactionDT // 86400`, `week = day // 7`, and `month = min(day // 30, 5)`, merging the 2–3 day tail stub into the final month (DECISIONS D-003). Relative time is all a chronological protocol requires.
 
Two properties of the stream shape everything downstream. First, it is **non-stationary**: the monthly fraud rate moves from 2.48% (month 0) to 4.04% (month 1) and back down to 3.47% (month 5), and transaction volume swings week to week (Figures 1–2). A model fitted once is a snapshot of a moving target. Second, the **label is operational, not predictive**: `isFraud` reflects Vesta's chargeback-linked process (downstream transactions of a flagged account or card can inherit the label), and reporting delay censors the newest data. Together these make the argument for chronological evaluation and against any feature that aggregates the label. All modeling uses the labeled training file only; Kaggle's unlabeled test set appears nowhere in any evaluation claim.
 
![Figure 1 — class balance and amount distributions](reports/figures/figure_01_class_balance_amounts.png)

**Figure 1 states the two facts that shape every decision in Methods**  On the left, fraud is 3.5% of 590,540 transactions, which means a model that approves everything is 96.5% accurate while catching nothing — this is why we never report accuracy, why the headline ranking metric is PR-AUC rather than ROC-AUC (§3.3), and where the gradient-boosted model's positive-class weight of roughly 27 comes from, since that is just the ratio 96.5 : 3.5 (§3.4). On the right, the amount distributions for fraud and legitimate transactions sit almost on top of each other: medians of $75 and $68, a seven-dollar gap against a spread running from under a dollar to over ten thousand. Amount alone does not separate the two classes. That is worth pausing on, because the common instinct is to screen large transactions harder — and the data says large transactions are not especially fraudulent, while the threshold derivation in §3.6 says the amount cancels out of the decision regardless. The one place the curves genuinely diverge is at the bottom of the range, a small excess of fraud below a dollar, consistent with card testing: attackers verifying stolen numbers with charges too small for anyone to notice.
 
![Figure 2 — weekly volume and fraud rate, with the temporal split](reports/figures/figure_02_volume_fraud_over_time.png)

**Figure 2 is the reason the protocol is chronological rather than random.** The top panel is weekly volume, settling near 20,000 transactions a week after a heavier opening month; the short final bar is not a data problem but the dataset's last day, which falls alone in week 26. The bottom panel is the one that drives the design. The fraud rate is not a constant -it moves between roughly 2% and 5% across the window, with no trend stable enough to extrapolate, so the quantity the model is trying to predict is itself changing while the model predicts it. The shading shows what we do about that: train, validation, and test are contiguous blocks of time, so the model is only ever scored on weeks that came after everything it learned from (§3.1). And the test block is not a gentle one. The fraud rate climbs across it and ends above where validation left off, which means every threshold and retraining cadence fixed in §3.6 and §3.8 is being asked to hold up on a month that got harder, not easier.


## 3. Methods

This section details the pipeline we use. Four published components are applied in these explanations, each introduced where it's used: a fixed operating point from the Bank Account Fraud benchmark (§3.2), two standard calibration repairs (§3.5), Elkan's cost threshold (§3.6), and the savings convention of Bahnsen et al. (§3.7). The intuitive breakdown of this section is a four-step process: **class-weighted training distorts the probability scale on purpose (§3.4), calibration repairs it (§3.5), a threshold derived from error costs consumes the repaired probabilities (§3.6), and a dollar-denominated convention prices the resulting policy (§3.7).** Sections 3.1–3.3 detail the protocol, the metrics, and the features that the process is based upon.

### 3.1 Temporal protocol

![Figure P — the fixed temporal protocol](reports/figures/figure_00_protocol.png)

The dataset does not say when anything happened. In place of calendar timestamps, each transaction carries a single column, TransactionDT, holding one number: how many seconds had elapsed at that moment since a fixed starting point Vesta never disclosed. The smallest value in the file is 86,400 and the largest is 15,811,131, so the data cover about 15.7 million seconds — 182 days — but which 182 days, in which year, is unknown.

That anonymization costs two specific things. We cannot align the data to outside events, so a spike in one week cannot be attributed to a holiday, a retailer's sale, or a publicized breach. And we cannot claim any seasonal pattern, because we do not know the season. What survives is order and spacing: which transaction came first, and how many seconds separated any two. Those two facts are enough for everything this paper does, because splitting data chronologically requires knowing the sequence of events, not their dates.

From that single column we derive three coarser units of time. Dividing by 86,400 — the number of seconds in a day — converts the offset into a day number: day = TransactionDT // 86400, where // divides and discards the remainder. Weeks follow as week = day // 7. Months are month = min(day // 30, 5), treating a month as a flat 30 days.

The min(..., 5) handles a leftover. Day numbers run from 1 to 182, so thirty-day months cover days 1–29 as month 0, days 30–59 as month 1, and so on through days 150–179 as month 5 — leaving days 180, 181 and 182 to fall into a seventh month three days long. Evaluating on a three-day stub would be meaningless, so the min folds those days back into month 5. The consequence is that the first month is a day short and the last is three days long, which is the "tail merged into the final month" recorded in DECISIONS D-003.

The six months are then given three non-overlapping roles (Figure P). Months 0–3 are training (410,601 rows) — the only data any model's parameters ever see. Month 4 is validation (85,303 rows), where every choice requiring held-out data is made, once: calibrators are fitted, thresholds are selected, and the final model is chosen. Month 5 is test (94,636 rows), evaluated only after everything upstream is frozen, so no reported number can have been tuned toward it even accidentally.

The split runs along the calendar rather than at random because the deployed task does. A shuffled split trains on part of March and tests on the rest of March; a deployed model is never in that position, since it always scores a stretch of time that had not happened when it was fitted. The gap between those two situations is not merely philosophical — §4.1 measures how many points of performance it silently adds.

Hyperparameters — the settings fixed before fitting begins, such as how many leaves a tree may grow — are selected inside the training window alone. The procedure is GroupKFold with month as the group: the four training months are divided into four folds along month boundaries, each candidate configuration is fitted on three of them and scored on the fourth, and the folds rotate so every month serves once as the held-out one. Because the folds are whole months rather than random rows, each internal test is itself a prediction forward in time — a small rehearsal of the real task, run four times.

Both properties are checked by automated test rather than trusted. make test verifies that no transaction appears in more than one split and that no month appears in more than one fold.

### 3.2 What we measure

Two different questions get asked of this system, and conflating them is much of what this paper is about. The first is whether the model orders transactions well: does it place fraud above legitimate activity? The second is whether decisions built on those scores save money. This subsection covers the ordering metrics, because §3.4's tuning is scored on one of them. The money metrics require the decisions mechanisms and wait until §3.7.

Any fixed cutoff produces a set of flagged transactions, and two quantities describe it. Precision is the share of flagged transactions that really are fraud: flag 100, and if 30 are fraud, precision is 0.30. Recall, also called the true positive rate, or TPR, is the share of all fraud that got flagged: if 200 frauds occurred and 60 were caught, recall is 0.30. The two trade against each other. Lowering the cutoff flags more transactions, which catches more fraud but sweeps in more legitimate orders, so recall rises while precision falls. Sweeping the cutoff across its entire range traces out the precision–recall curve, and the area beneath that curve, PR-AUC, condenses performance at every possible cutoff into one number.

We report PR-AUC in preference to the more familiar ROC-AUC because of how rare fraud is here. ROC-AUC's second axis is the false positive rate (the share of legitimate transactions wrongly flagged) and with 96.5% of the data legitimate, that denominator is enormous. A model can wrongly flag several thousand good customers and barely disturb its false positive rate, so ROC-AUC remains high and reassuring in circumstances a fraud team would consider a crisis. PR-AUC's denominator is the flagged set itself, which is small, so those same false flags register immediately. Both are reported in §4.1; PR-AUC is the one to watch.

The single headline number is one point on that trade-off rather than the whole curve: TPR at 5% FPR, adopted from the Bank Account Fraud benchmark, which standardizes evaluation of drifting, imbalanced tabular data at a fixed operating point,c"we select the threshold in order to obtain 5% false positive rate (FPR), and measure the true positive rate (TPR) at that point" [11]. In plain terms, each model's cutoff is tuned until it wrongly flags exactly 5% of legitimate transactions, and we then ask what share of the real fraud it caught. A curve is the right summary of a single model, but two models can only be compared once both are set to the same strictness, and this fixes the strictness. It also matches the real constraint on a fraud operation: review capacity is finite, that capacity determines how many good customers may be interrupted, and models compete on how much fraud they catch within that allowance.

### 3.3 Features

Some features are computed from one transaction on its own, with no reference to any other row. We use the logarithm of the amount (which compresses a long-tailed range, so the step from $10 to $100 counts as much as the step from $100 to $1,000), the cents portion of the amount (whether it ends in .00 or in something like .37, round and odd amounts arise from different processes, and currency conversion leaves a characteristic remainder), the hour of day, and the day of week.

The remaining engineered features summarize an entity's own past. Two entity keys are used: the card identifier card1, and the pair (card1, addr1) —a card together with its billing address. For each key, at each transaction, we compute how many transactions that entity has made before now, how many seconds have passed since its previous one, its average amount so far, and the ratio of the current amount to that average. "So far" is meant literally: the average at a given row includes every earlier transaction by that entity and no later one, so the value shifts as the entity's history accumulates. That is what expanding means here — the window grows forward from the beginning rather than sliding along a fixed length. These are behavioral-consistency signals; they ask whether this transaction resembles this card's own history rather than whether it resembles fraud in general.

We call such features causal in a narrow and specific sense: a feature is causal when its value at any row could have been computed on the day that row occurred, using only rows that had already happened. This is a statement about what information was available when, not a claim about causation in the statistical sense.

The property is verified rather than assumed. tests/test_causal_features.py truncates the dataset at row i, recomputes every aggregate on that shortened copy, and checks that the value at row i matches the value computed on the complete dataset. If deleting the future changes a feature's present value, that feature was reading the future, and the test fails. This discipline is a direct response to the precedent described in §1: entity-history features are powerful precisely because they let a model recognize a card it has encountered before, which is also what makes careless versions of them detach leaderboard scores from production reality.

No aggregate uses the fraud label. In production the label is a chargeback that arrives weeks after the transaction, so a feature such as "this card's past fraud rate" assumes information that does not exist at the moment the decision must be made — leakage of the operational kind described in §2. None is used here.

### 3.4 Models and class imbalance

We evaluate two models, named for the roles they would play in production. The champion is the incumbent that a new model must beat to justify replacing it; the challenger is the model attempting to.

The champion is L2-regularized logistic regression on the interpretable feature blocks (core fields, counts, time deltas, match flags, and numeric identity). Its preprocessing is fitted on the training window only, which matters because each step involves a quantity learned from data. Missing values are filled with the median of that column as measured in the training months. Each column is rescaled to comparable units using the training months' center and spread, so that a feature measured in dollars and one measured in days contribute on similar footing. The D block's time-since-last-event fields are frequently missing for a substantive reason — a card with no prior history has no "days since previous transaction" — so each receives a companion column recording whether the value was absent, letting the model treat the absence itself as information. Categorical fields are expanded into one column per value, a one-hot encoding, restricted to the 30 most frequent values in the training months so that values first appearing later cannot introduce new columns at test time.

The challenger is LightGBM, a gradient-boosted tree ensemble, run on all feature blocks with its native handling of categorical fields, which requires no one-hot expansion. The champion's role is to set the floor a black box must clear to be worth its opacity.

Fraud is 3.5% of rows. A model minimizing average error on that mixture can score well by calling nearly everything legitimate, so the rare class needs amplifying. We do this by weighting rather than resampling: scale_pos_weight is set to the ratio of legitimate to fraudulent rows in whatever window is being fitted — roughly 27 to 1 — which instructs the training procedure to count each fraudulent transaction as though it were 27 transactions. The rare class then influences the fitted model in proportion to its importance rather than its frequency.

This correction has a deliberate side effect, and it is the hinge of the entire section. A model trained as though fraud were 27 times more common than it is produces scores appropriate to that imagined world rather than the real one: its numbers run systematically high. The ordering is unharmed — inflating one class uniformly does not change which transactions look more suspicious than which — so every metric in §3.2 is unaffected. But the numbers can no longer be read as probabilities, and the decision rule in §3.6 reads them as exactly that. Section 3.5 is the repair.

We avoid SMOTE and other synthetic oversampling for a related reason: resampling manufactures a training distribution that never existed, distorts the same probability scale in a less controllable way, and interacts poorly with post-hoc calibration. Weighting plus explicit recalibration achieves the same control over the operating point while keeping the probability semantics auditable (DECISIONS D-004/D-005).

Hyperparameter search is deliberately modest — Optuna's TPE sampler, 12 trials within a 45-minute budget, scored by PR-AUC on the GroupKFold-by-month folds of §3.1, with a best fold score of 0.626. This project spends its rigor on evaluation rather than tuning depth.

### 3.5 From scores to probabilities

A model's scores are calibrated when they can be read as frequencies: among all transactions scored 0.20, about one in five should actually turn out to be fraud. Ranking well does not deliver this, because a model can order transactions flawlessly while its numbers run uniformly high or low, and ordering is what most training procedures optimize. In this case the numbers are known to run high, by construction, from the weighting in §3.4. Guo et al. document that modern learners are miscalibrated as the rule rather than the exception, and that simple corrections applied afterward largely repair them [12c].

The repair is a second, much smaller model fitted after the first: a function that takes the raw score in and returns a corrected probability out. It is fitted on month 4 and never on training or test data, because it must observe how scores behave on transactions the main model did not learn from. Two standard forms are tried. Platt scaling fits a two-parameter S-shaped curve, which is stable on limited data but assumes the distortion has that shape [12a]. Isotonic regression fits a staircase whose only constraint is that it never steps downward, which assumes nothing about shape but needs more data to fit reliably [12b]. Both are monotone, meaning neither can ever swap the order of two transactions, so every ranking metric in §3.2 is mathematically unchanged by calibration and only the meaning of the numbers moves. Whichever calibrator achieves the lower Brier score on validation — the average squared difference between the predicted probability and the actual outcome, treating the outcome as 1 for fraud and 0 otherwise, with lower being better — is frozen and carried forward to test.

Calibration also decays. A correction fitted on month 4 is fitted to month 4's particular mixture of customers and attacks, and that mixture moves. So calibration quality is measured on future data rather than assumed, weekly, using the Brier score and expected calibration error: group transactions into bands of similar predicted probability, and within each band compare the average prediction against the fraction that were actually fraud; the average of those gaps is the ECE. Section 4.2 shows the repair on the test month (Figure 4), and §4.4 shows what elapsed time does to it.

### 3.6 From probabilities to decisions

A calibrated probability is still not a decision. The usual way to obtain one is to search: try many candidate cutoffs on held-out data and keep whichever performs best on some chosen metric. Elkan's result is that once the costs of the two errors are written down, no search is needed — the cost-minimizing cutoff follows from the costs alone [2].

The reasoning is a comparison of two expected losses, made one transaction at a time. Take a $100 order that the model scores at fraud probability p. Approving a fraud means writing off the amount: a $100 loss with probability p, so an expected loss of p × $100. Declining a legitimate order forfeits the margin on that sale — say 15% of it, $15 — which happens with probability 1 − p, so an expected loss of (1 − p) × $15. Whichever number is smaller names the cheaper action. At p = 0.10, approving expects $10 against declining's $13.50, so the order should be approved. At p = 0.20, approving expects $20 against declining's $12, so it should be declined. The ranking flips at p = 0.1304, where both sides come to $13.04. That crossing point is the threshold.

Two things about it deserve a pause. First, the transaction amount cancels out. Writing the false-decline cost as k times the amount — k = 0.15 in the example — the crossing point works out to k/(1+k), which is the same number for a $12 order and a $2,000 one. The instinct to apply extra scrutiny to large transactions misreads the trade-off: a larger amount scales both mistakes together, since it means both a larger write-off and a larger lost sale, so it never changes which mistake is worse. Second, the cutoff lands at 13%, nowhere near the intuitive 50%. A rule that declines whenever fraud is more likely than not is quietly asserting that the two errors cost the same amount. Here the miss costs nearly seven times the false decline, so the cutoff belongs well below half. There is no way to choose a cutoff without pricing the two errors against each other — only ways of doing it without noticing.

The threshold is not learned from data. It follows from one business judgment — what a false decline is worth relative to a missed fraud — and arithmetic. That judgment is arguable, so rather than defend a single figure, §4.3 reports a sensitivity grid across k ∈ {0.05, 0.15, 0.30, 0.60, 1.0}, spanning the range from "false declines are cheap" to "a false decline hurts as much as the fraud" (Figures 5 and 6). And the arithmetic is only as sound as the p it consumes. Handed a score of 0.20 by a model whose 0.20-scored transactions are in truth fraudulent 5% of the time, the rule declines orders it should approve, computing correctly all the way to the wrong answer. That is why §3.5 comes first, and why §4.3 prices the calibrated and uncalibrated variants as separate policies.

### 3.7 Pricing a policy

The threshold prices one decision; evaluating a system means pricing a month of them. Bahnsen, Aouada & Ottersten adapt cost-sensitive evaluation to card fraud, where the cost of a mistake belongs to the individual transaction rather than to the category of error — approving a $2,000 fraud is not the same mistake as approving a $12 one, so counting errors treats unlike things alike [10]. Their savings measure works by comparison against doing nothing: total the cost a policy's decisions incur, divide by the cost of the cheapest policy that requires no model at all — approve everything, or decline everything, whichever is cheaper on that data — and report the fraction of that cost removed, savings = 1 − Cost_policy / Cost_baseline. Zero means the model earned nothing over doing nothing. One means it removed all cost. A negative value means it made matters worse than having no model. The denominator is the whole point: it fixes an honest zero, so a model cannot appear impressive merely because fraud is rare.

Six policies are compared on identical test transactions: the two trivial policies; the F1-optimal cutoff, which maximizes the harmonic mean of precision and recall — a common statistical default that implicitly treats the two errors as equally costly; Youden's J, which maximizes TPR − FPR and is equally cost-blind; and the cost-derived threshold of §3.6 applied to uncalibrated and to calibrated probabilities.

Two rules keep the comparison fair. The F1 and Youden cutoffs are themselves selected on the calibrated month-4 scores, so §4.3 isolates the effect of the thresholding rule rather than handicapping the baselines with worse inputs. And every threshold is frozen before the test month is opened.

Uncertainty is reported by paired bootstrap. A single test month is one sample of the world, and a policy might look good on it partly through luck. To measure how much, we construct 1,000 synthetic test months by drawing transactions from the real one at random with replacement, so each synthetic month is the same size but a slightly different mixture, and score every policy on each. The spread of results across those 1,000 months gives the confidence interval. Paired means all policies are scored on the identical synthetic month before moving to the next, so when we report a difference between two policies, whatever luck a given resample carried affected both equally and cancels out of the difference. An interval on a difference therefore reflects the change in policy rather than the change in sample.

### 3.8 Drift and ablation design

Everything above evaluates a frozen snapshot. This subsection asks what elapsed time does to it.

Weekly performance is traced across months 4 and 5 under three retraining cadences: static (train once on months 0–3 and never update), expanding (retrain at each month boundary on all data available so far), and sliding (retrain each month on the trailing three months only). One rule keeps the three comparable: every training window holds out its own final 14 days to fit its calibrator (D-006), so no cadence enjoys a fresher calibrator than another and the comparison isolates the retraining schedule. Cadences, thresholds, and holdout rules are all fixed before the weekly clock starts, so the curves in §4.4 (Figure 7) involve no further choices.

Input drift is measured two ways (Figure 8). The population stability index (PSI) summarizes how far a single feature's distribution has moved. Divide the feature's range into bins and record what fraction of training-window rows fell into each; do the same for one evaluation week; then compare the two sets of fractions bin by bin. Matching shares produce a PSI near zero, and the more mass has migrated between bins, the larger it grows. We compute it per feature per week for the model's top 20 features by gain, LightGBM's accounting of how much each feature contributed to the splits it made.

Adversarial validation asks the same question from the opposite direction. Instead of examining features one at a time, it trains a separate classifier whose only job is to tell training-window rows apart from out-of-time rows. If the two populations are genuinely alike, that classifier cannot do better than guessing and its AUC sits near 0.5; the further above 0.5 it climbs, the more the inputs have actually shifted. We run it twice, with and without the expanding aggregates of §3.3, because entity counters grow mechanically as time passes — a card's transaction count can only increase — which would let the classifier succeed for a reason that has nothing to do with meaningful drift.

Ablation asks what each feature block is worth. Hyperparameters and tree count are held fixed throughout, so differences between configurations measure the features rather than tuning luck. The challenger is retrained under forward addition (block A alone, then A+B, and onward through A+…+F, where A is the core fields together with §3.3's aggregates and F is the identity table; §4.5 tabulates the lettering) and under leave-one-block-out, which removes a single block from the full set to see what its absence costs. Each configuration is scored on validation by its change in TPR@5%FPR and in savings, and the selected configuration is confirmed exactly once on test (§4.5, Figure 9).

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
