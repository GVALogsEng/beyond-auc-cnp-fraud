# DECISIONS.md — Beyond AUC: CNP Fraud Detection

Running log of judgment calls, per project spec. Format: ID, date (UTC), decision, rationale, alternatives considered.

---

## D-011 — 2026-07-16 — Completion + verification pass

**Status:** all ten stages executed end-to-end on the verified official archive.
Tests 18/18 green (including real-data split-integrity and prefix-causality
checks). A final audit script asserted all 71 numeric claims in README.md
against their persisted `results/metrics/*.json` sources: zero mismatches.
Citations: 10 anchors verified at build time with working links and quoted
claims; 2 spec anchors dropped for failing verification (recorded in
`results/metrics/citation_verification.json` and README ref [9]).
One post-hoc statistic was added to `stage4_policies_test.json` after the
test pass: paired bootstrap CIs for savings *differences* (cal−uncal,
cal−F1) over the same resamples — a derived summary of already-frozen
policies, involving no new selection. Wall-clock: ≈4 h 45 m on 2 vCPUs
(measured per stage in README §7).

---

## D-000 — 2026-07-15 — Environment inventory at Stage 0 gate

**Decision:** Record the execution environment before any project work.

**Facts observed (cloud sandbox):**
- Python 3.11.15, git 2.43.0, 2 vCPUs, 7.8 GB RAM, ~30 GB free disk.
- Preinstalled and importable: pandas 3.0.2, numpy 2.4.4, scikit-learn 1.8.0, scipy 1.17.1, matplotlib 3.10.9, seaborn 0.13.2, joblib 1.5.3.
- Missing and currently uninstallable: lightgbm, xgboost, shap, streamlit, optuna, pytest, nbclient, nbconvert, ipykernel, kaggle (CLI).
- `ANTHROPIC_API_KEY` absent → Stage 9 will use the template-based fallback (per spec gating).
- No `~/.kaggle/kaggle.json`, no `KAGGLE_*` env vars.

---

## D-001 — 2026-07-15 — Stage 0 gate FAILED → halt and report (hard blocker)

**Decision:** Halt all project work at the Stage 0 gate and deliver a blocker report (`BLOCKER_REPORT.md`). No scaffold, code, or analysis built beyond this log and the report.

**Rationale:** Two independent prerequisite failures, both matching the spec's hard-blocker definition:

1. **Kaggle competition access unavailable** — spec hard blocker (a). No credentials anywhere in the environment, and network-level verification shows kaggle.com is blocked by the sandbox egress proxy regardless of credentials (HTTP 403 from proxy; API endpoint unreachable). `kaggle competitions files ieee-fraud-detection` cannot be run: the `kaggle` package itself cannot be installed.
2. **Package registries fully unreachable** — PyPI (pypi.org, files.pythonhosted.org), npm, GitHub, and Ubuntu archive all return 403/blocked, both direct and forced through the local egress proxy. Verified 2026-07-15 19:27–19:33 UTC with repeated attempts. The spec's core stack (LightGBM/XGBoost, SHAP, Streamlit, pytest, notebook execution deps) cannot be installed from inside the container.

**Alternatives considered and rejected:**
- *Unofficial mirrors of IEEE-CIS (HuggingFace/GitHub copies):* rejected. Provenance and checksums unverifiable, conflicts with Kaggle competition-rules compliance and the spec's data-hygiene clause; spec says "do not improvise around" blocker (a).
- *Substitute stack (sklearn HistGradientBoosting for LightGBM, unittest for pytest, drop SHAP/Streamlit):* rejected. The spec names LightGBM/XGBoost, TreeSHAP, Streamlit, and the pytest test files as non-negotiable deliverables; a substituted stack is a different project, not a defensible default.
- *Build lightgbm/xgboost/shap from GitHub source:* rejected. GitHub is also blocked from the container; and shap's build chain (numba/llvmlite) plus streamlit's (pyarrow) are not source-buildable here in any reasonable time.
- *Wait and retry (transient-outage hypothesis):* rejected. Failures were consistent across ~6 minutes, multiple hosts, and both network paths; the 403s are proxy policy responses, not timeouts.

**Chosen unblock path (keeps spec fully intact):** Geoff supplies, via the desktop app's "Add folder", a single folder containing (1) the official `ieee-fraud-detection.zip` downloaded under his own Kaggle account with competition rules accepted, and (2) Linux/cp311 wheels for the missing packages fetched with `pip download` on his Mac. The device-bridge file channel works independently of the container's blocked egress; `pip install --no-index --find-links` works offline. Exact commands in `BLOCKER_REPORT.md`.

---

## D-003 — 2026-07-15 — Trailing-stub month merge

**Decision:** `month = min(day // 30, 5)`. The 182-day span leaves a 2-3 day
stub at index 6; it is merged into month 5 (test) so the test window stays
contiguous and every month index has substantial volume. Weekly decay curves
likewise merge a trailing stub week into the preceding week when it has under
half the median weekly volume.

**Alternatives:** dropping the stub days (wastes labeled data); keeping index 6
(a 3-day "month" would distort GroupKFold and monthly statistics).

## D-004 — 2026-07-15 — No label-based aggregation features

**Decision:** entity aggregates use counts, timing, and amounts only — never
`isFraud`. Chargeback labels arrive weeks late in production; past-label
features computed from the same table would assume instant label availability
and silently inflate all metrics. This is logged as a deliberate divergence
from many leaderboard solutions.

## D-005 — 2026-07-15 — Champion LR feature scope and encodings

**Decision:** the logistic-regression champion uses blocks A-D plus numeric
identity features (V1-V339 excluded; DeviceInfo excluded as ~1,800-level
categorical); one-hot top-30 categories fit on the fitting window only;
medians/scaler moments likewise window-fit. The challenger uses all blocks.
Rationale: the champion is the interpretable benchmark, not a feature-parity
twin; both models see identical *rows* under every split regime, which is what
the optimism-gap comparison requires. C grid {0.03, 0.1, 0.3} selected on an
inner temporal split (train months 0-2, select on month 3) — modest by design.

## D-006 — 2026-07-15 — Uniform retraining protocol for the drift stage

**Decision:** in Stage 5, every policy (static/expanding/sliding) trains on its
window minus the final 14 days and calibrates on those 14 days. This differs
from the headline model (trained on all of months 0-3, calibrated on month 4)
by design: the drift stage needs the three policies to be comparable under one
rule, including at month 5 where expanding/sliding have no untouched month-4
holdout. Static and expanding coincide at month 4 by construction; the curves
show this honestly rather than manufacturing a difference.

## D-007 — 2026-07-15 — Notebooks are thin views over persisted artifacts

**Decision:** the six numbered notebooks load `results/metrics/*.json` and
display the persisted figures; they do not retrain models. Reproduction is
`make data features train evaluate figures`; notebooks are the annotated read
of those artifacts. This guarantees notebook prose can never diverge from the
numbers the paper reports, and keeps `make notebooks` (generate + execute)
under a minute.

## D-008 — 2026-07-15 — XGBoost omitted from the executed pipeline

**Decision:** proceed with LightGBM only (spec names it the primary
challenger). The 132 MB xgboost wheel repeatedly failed the device-bridge
transfer (silent drop above ~100 MB); rather than spend Geoff's time splitting
an optional dependency, it is omitted. `requirements.txt` pins the executed
environment; adding xgboost changes no interface.

## D-009 — 2026-07-16 — Synthetic smoke test before the real run

**Decision:** while blocked on the data transfer, the full pipeline was
exercised end-to-end in a scratch copy (`/tmp/smoke`, outside the repo) on a
synthetic IEEE-CIS-shaped dataset (60K rows, same schema). Nothing from that
run touches `results/`; its only purpose was to catch bugs before the real
2-3 h training run. Bugs found and fixed: (1) the champion's design-matrix
builder classified columns by name list, crashing on string identity columns
— now partitions by dtype at fit time and drops >60-cardinality strings;
(2) all-NaN-in-window numeric columns survived median imputation — medians
now fall back to 0; (3) an invalid logging format string. Also verified:
figures render, notebooks execute, Streamlit app boots (HTTP 200), and the
drift stage reproduces the designed static≡expanding identity in month 4.

**Data transfer note:** the device bridge silently dropped files >~100 MB;
the archive crossed as two 60 MB `split` parts and was reassembled with an
exact SHA-256 match against the checksum computed on the source machine
(4cc646da09d0...e0b829, recorded in results/metrics/data_checksums.json).

## D-010 — 2026-07-16 — Tuning budget enforcement on 2 vCPUs

**Decision:** the first live tuning run projected ~10 min/trial, and the stop
rule's 8-trial floor would have pushed the search to ~80 minutes — double the
45-minute budget. Training was stopped six minutes in and relaunched with:
budget-first stopping (minimum 4 trials, stop as soon as the projected next
trial exceeds the budget), trial evaluation on the two most-recent GroupKFold
month folds instead of all four (same GroupKFold-by-month mechanism the spec
fixes, evaluated on fewer splits), a 40% tuning row subsample, and early
stopping tightened to 75 rounds. Final refits still use full windows and the
grouped-CV regime still runs all four folds. Trade-off accepted: a slightly
noisier search signal in exchange for staying inside the spec's "modest
search" and runtime targets; the search space itself is unchanged.

## D-012 — 2026-07-16 — Revision pass at Geoff's request (PDFs + figure fixes)

**Decision:** post-delivery revision per owner feedback. (1) Figure defects
fixed: mathtext-mangled dollar signs in Fig 1's title; threshold labels
overlapping curves in Fig 5 (moved above the axes); the diverging heatmap
norm washing out positive cells in Fig 6 (TwoSlopeNorm); legend-on-data and
annotation collisions in Figs 7-8; SHAP waterfalls showing categorical codes
instead of real values (e.g. "16 = P_emaildomain" -> "gmail.com ="), plus an
explicit "top calibration step" note where isotonic maps to exactly 1.0.
(2) The ASCII temporal-protocol diagram replaced by a rendered figure
(figure_00_protocol.png). (3) New `make pdfs` target: markdown -> HTML
(mistune) -> headless Chromium print-to-PDF (offline; reportlab unavailable
in the sandbox). Produces the paper, the run report, and two plain-English
companion documents (reports/companions/) as typeset PDFs with embedded
figures. No result, metric, or figure *data* changed in this pass.

## D-002 — 2026-07-15 — Pre-logged adaptations, pending unblock

**Decision:** Three small adaptations will apply once data/wheels arrive; logged now for transparency.

1. **`make data` behavior:** inside this sandbox it will verify and unpack a pre-placed `ieee-fraud-detection.zip` (with recorded SHA-256) instead of calling the Kaggle API. The README's reproducibility section will document the standard `kaggle competitions download -c ieee-fraud-detection` path for anyone reproducing on a normal machine, and the Makefile will support both (API if available, else local archive). Raw data stays gitignored either way.
2. **Stage 9:** no `ANTHROPIC_API_KEY` in this environment → build the template-based narrative fallback and the (unexercised) API code path, exactly as the spec's gate prescribes.
3. **Compute:** 2 vCPUs. Hyperparameter search will sit at the modest end of the spec's allowance (small grid / early-stopped trials) to respect the ~2–3 h runtime target; if the full pipeline overshoots on 2 cores, actual wall-clock will be reported honestly in the README's reproducibility section.
