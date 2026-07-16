"""Generate the six thin, numbered notebooks.

Design (DECISIONS.md D-007): notebooks are *views* over persisted pipeline
artifacts -- they load results/metrics/*.json and display the corresponding
figures. Reproduction of the numbers happens through the Makefile targets;
this keeps every notebook runnable in seconds and guarantees the notebook
narrative can never diverge from the persisted results.

`python -m src.visualization.make_notebooks --execute` writes and executes
all notebooks in order with nbclient.
"""
from __future__ import annotations

import argparse
import json

import nbformat as nbf

from src import config

NB_DIR = config.ROOT / "notebooks"

HEADER = ("*This notebook is a thin view: it renders artifacts persisted by "
          "the pipeline (`make data features train evaluate figures`). "
          "No result is computed here.*")


def _nb(cells) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb["metadata"]["kernelspec"] = {"name": "python3", "language": "python",
                                    "display_name": "Python 3"}
    nb["cells"] = cells
    return nb


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(src):
    return nbf.v4.new_code_cell(src)


SETUP = """\
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))
import pandas as pd
from IPython.display import Image, display

RESULTS = Path.cwd().parent / "results" / "metrics"
FIGURES = Path.cwd().parent / "reports" / "figures"

def load(name):
    with open(RESULTS / f"{name}.json") as fh:
        return json.load(fh)

def fig(name):
    display(Image(str(FIGURES / f"{name}.png"), width=880))
"""


def nb01():
    return _nb([
        md(f"# 01 - EDA and leakage audit\n\n{HEADER}"),
        code(SETUP),
        md("## Dataset shape, class balance, identity coverage"),
        code("eda = load('stage1_eda')\n"
             "{k: eda[k] for k in ['n_rows','n_cols','fraud_rate','n_fraud',"
             "'identity_coverage','span_days']}"),
        code("fig('figure_01_class_balance_amounts')"),
        md("## Temporal structure\n\nVolume and fraud rate move week to week; "
           "the last two months are visibly different from the first four. "
           "Any evaluation that shuffles these rows together is answering a "
           "question production never asks."),
        code("fig('figure_02_volume_fraud_over_time')"),
        code("pd.DataFrame(eda['monthly'])"),
        md("## Missingness by feature block"),
        code("pd.Series(eda['missingness_by_block']).sort_values()"
             ".map('{:.1%}'.format)"),
        md("""## Leakage audit (why random CV is invalid here)

1. **Temporal structure.** Transactions are a time series of adversarial
   behavior; random K-fold trains on the future and validates on the past.
2. **D-features.** `D1`-`D15` are time deltas anchored to calendar events;
   shuffled folds let the model interpolate between past and future values of
   the same underlying clock.
3. **Entity aggregations.** Frequency/velocity features accumulate per card
   profile. Computed naively (full-table counts) they encode the future;
   computed causally they still correlate strongly across a card's
   transactions, so shuffled folds place the same entity on both sides of the
   split. The 1st-place IEEE-CIS solution (Deotte & Yakovlev) made exactly
   this point: their UID-level features required GroupKFold-by-month to
   validate honestly -- client-level memorization otherwise masquerades as
   generalization.
4. **Labels.** `isFraud` propagates to downstream transactions of a flagged
   account, so a shuffled split can leak one fraud episode across folds.

Consequences for this project: chronological splits everywhere, GroupKFold
by month for model selection, expanding-window (strictly past) aggregations
enforced by `tests/test_causal_features.py`, and no label-based aggregation
features at all (chargeback labels arrive weeks late in production)."""),
    ])


def nb02():
    return _nb([
        md(f"# 02 - Baselines and the optimism gap\n\n{HEADER}"),
        code(SETUP),
        md("## Tuning (GroupKFold by month, modest Optuna budget)"),
        code("t = load('stage2_lgbm_tuning')\n"
             "{k: t[k] for k in ['best_params','best_cv_pr_auc','n_trials',"
             "'elapsed_s']}"),
        md("## The same models under three split regimes"),
        code("gap = load('stage2_optimism_gap_final')\n"
             "rows = []\n"
             "for model in ('lgbm','lr'):\n"
             "    for regime in ('random_cv_mean','grouped_cv_mean','test'):\n"
             "        r = dict(gap[model][regime]); r['model']=model; r['regime']=regime\n"
             "        rows.append(r)\n"
             "pd.DataFrame(rows).set_index(['model','regime']).round(4)"),
        code("pd.DataFrame({m: gap[m]['gap_random_minus_test'] for m in ('lgbm','lr')}"
             ").round(4)"),
        code("fig('figure_03_optimism_gap')"),
        md("Random 5-fold CV reports a number that the deployed model never "
           "achieves out-of-time; the gap is the optimism a leaderboard-style "
           "evaluation silently adds."),
    ])


def nb03():
    return _nb([
        md(f"# 03 - Calibration\n\n{HEADER}"),
        code(SETUP),
        code("cal = load('stage3_calibration')\n"
             "pd.DataFrame(cal['val_metrics']).round(5)"),
        code("calt = load('stage3_calibration_test')\n"
             "pd.DataFrame(calt['test_metrics']).round(5)"),
        code("fig('figure_04_reliability')"),
        md("Class-weighted training deliberately distorts probabilities "
           "(scale_pos_weight); calibration on the validation month repairs "
           "them. The chosen method is selected by validation Brier score "
           "and frozen before the test month is touched."),
        code("cal['chosen_method'], cal['platt_coef']"),
    ])


def nb04():
    return _nb([
        md(f"# 04 - The cost-sensitive decision layer\n\n{HEADER}"),
        code(SETUP),
        md("Decline iff `p > k/(1+k)` (Elkan 2001): with FN cost = amount and "
           "FP cost = k x amount, expected cost of declining beats approving "
           "exactly when `p*A > (1-p)*k*A`. Amount cancels -- the threshold is "
           "amount-independent, but only meaningful on calibrated p."),
        code("pol = load('stage4_policies_test')\n"
             "tbl = pd.DataFrame({k: v for k, v in pol['policy_table_test'].items()"
             " if not k.startswith('_')}).T\n"
             "tbl[['total_cost','savings','fraud_dollars_caught',"
             "'legit_dollars_declined','tpr','fpr']].round(4)"),
        code("pd.DataFrame(pol['bootstrap_ci']).T.round(4)"),
        code("pol['calibration_dollar_gap']"),
        code("fig('figure_05_cost_vs_threshold')"),
        code("fig('figure_06_sensitivity_heatmap')"),
    ])


def nb05():
    return _nb([
        md(f"# 05 - Drift and retraining cadence\n\n{HEADER}"),
        code(SETUP),
        code("drift = load('stage5_drift')\n"
             "pd.DataFrame(drift['weekly']['static']).round(4).head(10)"),
        code("fig('figure_07_decay_curves')"),
        code("psi = load('stage5_psi')\n"
             "pd.Series(psi['max_psi_per_feature']).sort_values(ascending=False)"
             ".head(10).round(3)"),
        code("adv = load('stage5_adversarial')\n"
             "pd.DataFrame(adv['internal_oot'])[['tag','cv_auc_mean','cv_auc_sd']]"),
        code("fig('figure_08_psi_adversarial')"),
    ])


def nb06():
    return _nb([
        md(f"# 06 - Ablation and explainability\n\n{HEADER}"),
        code(SETUP),
        code("abl = load('stage6_ablation')\n"
             "pd.DataFrame(abl['results']).T[['n_features','tpr_at_5fpr',"
             "'pr_auc','savings','delta_tpr_at_5fpr','delta_savings']].round(4)"),
        code("abl['selected_blocks'], abl['dropped_block']"),
        code("fig('figure_09_ablation')"),
        code("shap_summary = load('stage7_shap')\n"
             "pd.Series(shap_summary['mean_abs_shap_top30']).head(15).round(3)"),
        code("fig('figure_10a_shap_beeswarm')"),
        code("fig('figure_10b_waterfall_tp')"),
        code("fig('figure_10c_waterfall_fp')"),
    ])


BUILDERS = {"01-eda-leakage-audit": nb01,
            "02-baselines-optimism-gap": nb02,
            "03-calibration": nb03,
            "04-cost-decision-layer": nb04,
            "05-drift-retraining": nb05,
            "06-ablation-shap": nb06}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    NB_DIR.mkdir(exist_ok=True)
    for name, builder in BUILDERS.items():
        path = NB_DIR / f"{name}.ipynb"
        nbf.write(builder(), path)
        print(f"wrote {path}")

    if args.execute:
        from nbclient import NotebookClient
        for name in BUILDERS:
            path = NB_DIR / f"{name}.ipynb"
            nb = nbf.read(path, as_version=4)
            client = NotebookClient(nb, timeout=300,
                                    resources={"metadata": {"path": str(NB_DIR)}})
            client.execute()
            nbf.write(nb, path)
            print(f"executed {path}")


if __name__ == "__main__":
    main()
