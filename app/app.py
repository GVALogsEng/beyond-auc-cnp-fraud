"""Streamlit demo: cost-sensitive, calibrated CNP fraud decisioning.

Runs entirely from cached artifacts in app/artifacts/ (produced by
`make evaluate`); no Kaggle download, model, or network access needed at
app runtime. All interactive economics are computed live on the cached
stratified 20K-row test sample and labeled as such.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ART = Path(__file__).resolve().parent / "artifacts"

BLUE, GREEN, MAGENTA, ORANGE, INK, MUTED, GRID = (
    "#2a78d6", "#008300", "#e87ba4", "#eb6834", "#0b0b0b", "#898781", "#e1e0d9")

st.set_page_config(page_title="Beyond AUC - CNP fraud decisioning",
                   layout="wide")


@st.cache_data
def load_artifacts():
    sample = pd.read_parquet(ART / "sample.parquet")
    shap_top = pd.read_parquet(ART / "shap_top.parquet")
    with open(ART / "policy_table.json") as f:
        policy = json.load(f)
    with open(ART / "drift.json") as f:
        drift = json.load(f)
    with open(ART / "calibration.json") as f:
        calib = json.load(f)
    with open(ART / "model_card.json") as f:
        card = json.load(f)
    with open(ART / "narratives.json") as f:
        narratives = {n["TransactionID"]: n for n in json.load(f)}
    return sample, shap_top, policy, drift, calib, card, narratives


def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(color=GRID, linewidth=0.6, axis="y")
    ax.set_axisbelow(True)
    return ax


def page_overview(card, policy):
    st.title("Beyond AUC: cost-sensitive, calibrated CNP fraud decisioning")
    st.caption("Model card - intended use, data, headline metrics, limitations")

    h = card["headline"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Savings @ k=0.15 (test)", f"{h['savings_at_k_0.15']:.1%}",
              help="1 - cost/baseline; 95% CI "
                   f"[{h['savings_ci95'][0]:.1%}, {h['savings_ci95'][1]:.1%}]")
    c2.metric("TPR @ 5% FPR (test)", f"{h['tpr_at_5fpr']:.1%}",
              help=f"95% CI [{h['tpr_ci95'][0]:.1%}, {h['tpr_ci95'][1]:.1%}]")
    c3.metric("PR-AUC (test)", f"{h['pr_auc']:.3f}")
    c4.metric("Test fraud rate", f"{policy['test_fraud_rate']:.2%}")

    st.subheader("Intended use")
    st.write(card["model"])
    st.write(f"**Data.** {card['data']}")
    st.write(f"**Evaluation protocol.** {card['protocol']}")
    st.subheader("Limitations")
    for lim in card["limitations"]:
        st.write(f"- {lim}")
    st.info(f"**LLM placement.** {card['cold_path_llm']}", icon=":material/psychology:")


def page_economics(sample, policy):
    st.title("Threshold economics")
    st.caption("Computed live on the cached stratified 20K test sample - "
               "headline paper numbers use the full test month.")

    col1, col2 = st.columns([1, 2])
    with col1:
        k = st.slider("False-decline cost multiplier k", 0.01, 1.0, 0.15, 0.01,
                      help="A false decline costs k x amount; a missed fraud "
                           "costs the full amount.")
        t_star = k / (1 + k)
        override = st.checkbox("Override analytic threshold", value=False)
        thr = st.slider("Decline threshold on calibrated p", 0.0, 1.0,
                        float(t_star), 0.005) if override else t_star
        st.metric("Analytic threshold k/(1+k)", f"{t_star:.3f}")

    y = sample["y"].to_numpy()
    amt = sample["amt"].to_numpy(dtype="float64")
    p = sample["p_cal"].to_numpy(dtype="float64")
    declined = p > thr

    fn_cost = amt[(y == 1) & ~declined].sum()
    fp_cost = k * amt[(y == 0) & declined].sum()
    cost = fn_cost + fp_cost
    baseline = min(amt[y == 1].sum(), k * amt[y == 0].sum())
    savings = 1 - cost / baseline

    with col2:
        m1, m2, m3 = st.columns(3)
        m1.metric("Total cost (sample)", f"${cost:,.0f}")
        m2.metric("Savings vs baseline", f"{savings:.1%}")
        m3.metric("Declined share", f"{declined.mean():.2%}")
        m4, m5, m6 = st.columns(3)
        m4.metric("Fraud $ caught", f"${amt[(y == 1) & declined].sum():,.0f}")
        m5.metric("Fraud $ missed", f"${fn_cost:,.0f}")
        m6.metric("Legit $ declined", f"${amt[(y == 0) & declined].sum():,.0f}")

    thrs = np.linspace(0.005, 0.995, 160)
    costs = [amt[(y == 1) & ~(p > t)].sum() + k * amt[(y == 0) & (p > t)].sum()
             for t in thrs]
    sav = 1 - np.array(costs) / baseline
    fig, ax = plt.subplots(figsize=(8, 3.2))
    style_ax(ax)
    ax.plot(thrs, sav * 100, color=BLUE, label="savings on cached sample")
    ax.axvline(t_star, color=MUTED, ls=":", lw=1)
    cur = 1 - (amt[(y == 1) & ~declined].sum()
               + k * amt[(y == 0) & declined].sum()) / baseline
    ax.scatter([thr], [cur * 100], color=ORANGE, zorder=5, s=60,
               label="current operating point")
    ax.set_xlabel("decline threshold")
    ax.set_ylabel("savings (%)")
    ax.legend(frameon=False)
    st.pyplot(fig, clear_figure=True)

    st.subheader("Frozen-policy comparison on the full test month")
    tbl = policy["policy_table_test"]
    rows = []
    for pol, r in tbl.items():
        if pol.startswith("_"):
            continue
        rows.append({"policy": pol, "total cost ($)": f"{r['total_cost']:,.0f}",
                     "savings": f"{r['savings']:.1%}",
                     "fraud $ caught": f"{r['fraud_dollars_caught']:,.0f}",
                     "legit $ declined": f"{r['legit_dollars_declined']:,.0f}",
                     "TPR": f"{r['tpr']:.1%}", "FPR": f"{r['fpr']:.2%}"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def page_inspector(sample, shap_top, narratives, policy):
    st.title("Transaction inspector")
    k = policy["k_central"]
    thr = k / (1 + k)

    subset = st.radio("Show", ["declined (flagged)", "fraud", "all"],
                      horizontal=True)
    view = sample
    if subset == "declined (flagged)":
        view = sample[sample["declined_central"]]
    elif subset == "fraud":
        view = sample[sample["y"] == 1]
    view = view.sort_values("p_cal", ascending=False).head(500)

    txn_id = st.selectbox("TransactionID (sorted by score, top 500)",
                          view["TransactionID"].tolist())
    row = sample[sample["TransactionID"] == txn_id].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Calibrated fraud probability", f"{row['p_cal']:.1%}")
    c2.metric("Decision @ k=0.15", "DECLINE" if row["p_cal"] > thr else "APPROVE")
    c3.metric("Amount", f"${row['amt']:,.2f}")
    c4.metric("Actual label", "fraud" if row["y"] == 1 else "legitimate")

    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("Top score contributions (TreeSHAP, log-odds)")
        contribs = shap_top[shap_top["TransactionID"] == txn_id] \
            .sort_values("shap", key=abs, ascending=True)
        fig, ax = plt.subplots(figsize=(6, 3.6))
        style_ax(ax)
        colors = [ORANGE if v > 0 else BLUE for v in contribs["shap"]]
        labels = [f"{f} = {v}" for f, v in
                  zip(contribs["feature"], contribs["value"].fillna("NA"))]
        ax.barh(labels, contribs["shap"], color=colors, height=0.62)
        ax.axvline(0, color=MUTED, lw=1)
        ax.set_xlabel("SHAP contribution to fraud log-odds")
        st.pyplot(fig, clear_figure=True)
    with right:
        st.subheader("Investigation narrative (cold path)")
        n = narratives.get(int(txn_id))
        if n:
            st.write(n["text"])
            st.caption(f"source: {n['source']} - narratives support analysts; "
                       "they never gate the authorization decision")
        else:
            st.caption("No narrative cached for this transaction (narratives "
                       "are generated for the top declined transactions).")
        st.subheader("Transaction fields")
        drop = ["declined_central"]
        st.dataframe(row.drop(labels=drop).to_frame("value"),
                     use_container_width=True)


def page_drift(drift, calib):
    st.title("Drift monitor")
    st.caption("Weekly out-of-time performance and input stability")

    colors = {"static": BLUE, "expanding": GREEN, "sliding": MAGENTA}
    metric = st.selectbox("Metric", ["tpr_at_5fpr", "pr_auc", "savings", "brier"])
    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    style_ax(ax)
    for pol, rows in drift["weekly"].items():
        ax.plot([r["week"] for r in rows], [r[metric] for r in rows],
                marker="o", ms=3.5, color=colors[pol], label=pol)
    ax.set_xlabel("week index")
    ax.set_ylabel(metric)
    ax.legend(frameon=False)
    st.pyplot(fig, clear_figure=True)

    st.subheader("Calibration stability (weekly Brier, out-of-time)")
    wk = calib["weekly_oot_calibration"]
    fig, ax = plt.subplots(figsize=(8.5, 3.0))
    style_ax(ax)
    for name, series, color in (("uncalibrated", wk["uncalibrated"], ORANGE),
                                ("calibrated", wk["calibrated"], BLUE)):
        ax.plot([r["week"] for r in series], [r["brier"] for r in series],
                marker="o", ms=3.5, color=color, label=name)
    ax.set_xlabel("week index")
    ax.set_ylabel("Brier score")
    ax.legend(frameon=False)
    st.pyplot(fig, clear_figure=True)


def main():
    sample, shap_top, policy, drift, calib, card, narratives = load_artifacts()
    page = st.sidebar.radio("Pages", ["Overview / model card",
                                      "Threshold economics",
                                      "Transaction inspector",
                                      "Drift monitor"])
    st.sidebar.caption(
        "Demo runs on a cached 20K stratified test sample; no Kaggle data or "
        "model needed at runtime. Built from the IEEE-CIS Fraud Detection "
        "dataset (Kaggle/Vesta).")
    if page == "Overview / model card":
        page_overview(card, policy)
    elif page == "Threshold economics":
        page_economics(sample, policy)
    elif page == "Transaction inspector":
        page_inspector(sample, shap_top, narratives, policy)
    else:
        page_drift(drift, calib)


if __name__ == "__main__":
    main()
