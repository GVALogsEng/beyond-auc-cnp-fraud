"""All paper figures, rendered exclusively from persisted artifacts.

Every panel reads results/metrics/*.json or data/interim arrays produced by
executed pipeline stages -- no figure recomputes a result. PNG @200dpi.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from src import config
from src.utils import get_logger, load_json
from src.visualization import style as S

log = get_logger("viz.figures")

POLICY_LABELS = {"approve_all": "Approve all", "decline_all": "Decline all",
                 "f1_opt": "F1-optimal", "youden": "Youden J",
                 "cost_uncal": "Cost thr.\n(uncalibrated)",
                 "cost_cal": "Cost thr.\n(calibrated)"}


# ---------------------------------------------------------------- fig 00 ----
def fig00_protocol():
    """The temporal protocol as a rendered timeline (replaces ASCII art)."""
    from matplotlib.patches import FancyArrowPatch, Rectangle as Rect

    fig, ax = plt.subplots(figsize=(9.2, 3.4))
    ax.set_xlim(-0.2, 6.55)
    ax.set_ylim(-2.75, 1.9)
    ax.axis("off")

    colors = {"train": "#cde2fb", "val": "#fff3d6", "test": "#fdeaea"}
    edge = {"train": S.C1_BLUE, "val": "#c98500", "test": "#e34948"}
    roles = [("train", range(0, 4)), ("val", [4]), ("test", [5])]

    # time arrow drawn first, hidden behind the opaque month boxes
    ax.annotate("", xy=(6.35, 0.425), xytext=(-0.15, 0.425), zorder=1,
                arrowprops=dict(arrowstyle="->", color=S.MUTED, lw=1.0))
    ax.text(6.38, 0.425, "time", fontsize=8.5, color=S.MUTED, va="center")

    for role, months in roles:
        for m in months:
            ax.add_patch(Rect((m + 0.03, 0.0), 0.94, 0.85, zorder=3,
                              facecolor=colors[role], edgecolor=edge[role],
                              linewidth=1.4))
            ax.text(m + 0.5, 0.425, f"month {m}", ha="center", va="center",
                    fontsize=9.5, color=S.INK, zorder=4)

    ax.text(2.0, 1.24, "TRAIN - months 0-3 (410,601 rows)", ha="center",
            fontsize=9.5, fontweight="semibold", color=S.C1_BLUE)
    ax.text(4.5, 1.24, "VALIDATION", ha="center", fontsize=9.5,
            fontweight="semibold", color="#9a6b00")
    ax.text(5.5, 1.24, "TEST", ha="center", fontsize=9.5,
            fontweight="semibold", color="#b22222")

    def brace(x0, x1, text, color, ytext, xtext=None, ha="center"):
        y = -0.22
        ax.plot([x0, x1], [y, y], color=color, lw=1.2)
        ax.plot([x0, x0], [y, y + 0.09], color=color, lw=1.2)
        ax.plot([x1, x1], [y, y + 0.09], color=color, lw=1.2)
        ax.plot([(x0 + x1) / 2, xtext if xtext else (x0 + x1) / 2],
                [y, ytext + 0.06], color=color, lw=0.7, alpha=0.55)
        ax.text(xtext if xtext else (x0 + x1) / 2, ytext, text, ha=ha,
                va="top", fontsize=8.3, color=S.INK_2, linespacing=1.5)

    brace(0.08, 3.94,
          "hyperparameter selection: GroupKFold, month = group\n"
          "causal features: expanding, strictly past-only aggregates",
          S.C1_BLUE, -0.50, xtext=2.0)
    brace(4.08, 4.94,
          "fit calibrators (Platt / isotonic)\n"
          "select thresholds + select model",
          "#9a6b00", -1.42, xtext=3.6)
    brace(5.08, 5.94,
          "touched exactly once,\nafter everything upstream is frozen",
          "#b22222", -1.42, xtext=5.51)

    ax.text(0.05, -2.45, "day = TransactionDT // 86400      week = day // 7      "
            "month = min(day // 30, 5)      182-day span; 2-3 day tail merged "
            "into month 5", fontsize=7.8, color=S.MUTED)
    ax.set_title("The fixed temporal protocol: every choice happens upstream "
                 "of the test month", fontsize=10.5, fontweight="semibold")
    return S.savefig(fig, "figure_00_protocol")


# ---------------------------------------------------------------- fig 01 ----
def fig01_class_balance():
    eda = load_json("stage1_eda")
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.1))

    ax = axes[0]
    frac = eda["fraud_rate"]
    ax.bar([0, 1], [1 - frac, frac], width=0.55,
           color=[S.C1_BLUE, S.C6_ORANGE], edgecolor="none")
    ax.set_xticks([0, 1], ["legitimate", "fraud"])
    ax.set_ylabel("share of transactions")
    for x, v in ((0, 1 - frac), (1, frac)):
        ax.text(x, v + 0.02, f"{v:.1%}", ha="center", color=S.INK, fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_title(f"Class balance (n = {eda['n_rows']:,})")

    ax = axes[1]
    h = eda["amount_hist"]
    edges = np.array(h["edges"])
    mids = np.sqrt(edges[:-1] * edges[1:])
    # light smoothing for readability (3-bin moving average; shapes unchanged)
    smooth = lambda v: np.convolve(v, np.ones(3) / 3, mode="same")
    ax.plot(mids, smooth(np.array(h["legit"])), color=S.C1_BLUE, label="legitimate")
    ax.plot(mids, smooth(np.array(h["fraud"])), color=S.C6_ORANGE, label="fraud")
    ax.set_xscale("log")
    ax.set_xlabel("TransactionAmt (USD, log scale)")
    ax.set_ylabel("density")
    ax.legend()
    med_l = h["quantiles"]["legit"]["0.5"]
    med_f = h["quantiles"]["fraud"]["0.5"]
    ax.set_title(f"Amounts (median legit \\${med_l:.0f}, fraud \\${med_f:.0f})")
    return S.savefig(fig, "figure_01_class_balance_amounts")


# ---------------------------------------------------------------- fig 02 ----
def fig02_volume_over_time():
    eda = load_json("stage1_eda")
    wk = pd.DataFrame(eda["weekly"])
    fig, axes = plt.subplots(2, 1, figsize=(8.6, 4.6), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1]})

    def shade(ax):
        for m0, m1, color, label in ((0, 4, None, "train (months 0-3)"),
                                     (4, 5, "#fff7e6", "validation (month 4)"),
                                     (5, 6.2, "#fdeaea", "test (month 5)")):
            if color:
                ax.axvspan(m0 * 30 / 7, m1 * 30 / 7, color=color, zorder=0)

    ax = axes[0]
    shade(ax)
    ax.bar(wk["week"], wk["n"], color=S.C1_BLUE, width=0.82)
    ax.set_ylabel("transactions / week")
    ax.set_title("Weekly volume and fraud rate; temporal split shaded "
                 "(white=train, amber=validation, red=test)")

    ax = axes[1]
    shade(ax)
    ax.plot(wk["week"], np.array(wk["fraud_rate"]) * 100, color=S.C6_ORANGE)
    ax.set_ylabel("fraud rate (%)")
    ax.set_xlabel("week index")
    return S.savefig(fig, "figure_02_volume_fraud_over_time")


# ---------------------------------------------------------------- fig 03 ----
def fig03_optimism_gap():
    gap = load_json("stage2_optimism_gap_final")
    metrics = [("pr_auc", "PR-AUC"), ("tpr_at_5fpr", "TPR @ 5% FPR")]
    regimes = [("random_cv_mean", "random 5-fold CV"),
               ("grouped_cv_mean", "GroupKFold (month)"),
               ("test", "out-of-time test")]
    colors = [S.C3_MAGENTA, S.C4_YELLOW, S.C1_BLUE]

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.3))
    width = 0.24
    for ax, (mkey, mlabel) in zip(axes, metrics):
        for j, (rkey, rlabel) in enumerate(regimes):
            vals = [gap["lgbm"][rkey][mkey], gap["lr"][rkey][mkey]]
            x = np.arange(2) + (j - 1) * width
            bars = ax.bar(x, vals, width=width - 0.02, color=colors[j],
                          label=rlabel)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.3f}",
                        ha="center", fontsize=6.8, color=S.INK_2)
        ax.set_xticks([0, 1], ["LightGBM", "Logistic reg."])
        ax.set_title(mlabel)
        lo = min(gap[m]["test"][mkey] for m in ("lgbm", "lr"))
        ax.set_ylim(max(0, lo - 0.18), None)
    axes[0].set_ylabel("score")
    axes[0].legend(loc="lower right")
    fig.suptitle("The optimism gap: random CV vs honest evaluation",
                 fontsize=10.5, fontweight="semibold", y=1.02)
    return S.savefig(fig, "figure_03_optimism_gap")


# ---------------------------------------------------------------- fig 04 ----
def fig04_reliability():
    val = load_json("stage3_calibration")
    test = load_json("stage3_calibration_test")
    chosen = val["chosen_method"]
    panels = [("uncalibrated", "Uncalibrated (test)", S.C6_ORANGE),
              (chosen, f"{chosen.capitalize()} (test)", S.C1_BLUE)]

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6), sharey=True)
    for ax, (key, title, color) in zip(axes, panels):
        rb = test["reliability_test"][key]
        mp = np.array([v for v in rb["mean_pred"] if v is not None])
        fp = np.array([v for m, v in zip(rb["mean_pred"], rb["frac_pos"])
                       if m is not None])
        cnt = np.array([c for m, c in zip(rb["mean_pred"], rb["count"])
                        if m is not None])
        ax.plot([0, 1], [0, 1], color=S.BASELINE, lw=1, ls="--", zorder=1)
        ax.scatter(mp, fp, s=np.clip(np.sqrt(cnt), 4, 26) * 3.2, color=color,
                   zorder=3, alpha=0.9)
        ax.plot(mp, fp, color=color, lw=1.2, zorder=2)
        m = test["test_metrics"][key]
        ax.set_title(f"{title}\nBrier {m['brier']:.4f} - ECE {m['ece']:.4f}")
        ax.set_xlabel("mean predicted probability")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
    axes[0].set_ylabel("observed fraud rate")
    fig.suptitle("Reliability before/after calibration (fit on validation month only)",
                 fontsize=10.5, fontweight="semibold", y=1.04)
    return S.savefig(fig, "figure_04_reliability")


# ---------------------------------------------------------------- fig 05 ----
def fig05_cost_curves():
    pol = load_json("stage4_policies_test")
    thr = load_json("stage4_thresholds")
    k = pol["k_central"]
    t_star = k / (1 + k)

    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    peak = -np.inf
    for key, label, color in (("sweep_uncal_test", "uncalibrated scores", S.C6_ORANGE),
                              ("sweep_cal_test", "calibrated scores", S.C1_BLUE)):
        sw = pol[key]
        sav = np.array(sw["savings"]) * 100
        peak = max(peak, sav.max())
        ax.plot(sw["thresholds"], sav, color=color, label=label)
    ax.set_ylim(-30, peak + 6)   # decline-all plunge continues off-axis
    # threshold markers: vertical lines with labels ABOVE the axes (no overlap
    # with data); two stagger rows because Youden and k/(1+k) sit close.
    ax.axvline(t_star, color=S.INK_2, lw=1, ls=":")
    ax.annotate(f"k/(1+k) = {t_star:.3f}", xy=(t_star, 1.0),
                xycoords=("data", "axes fraction"), xytext=(t_star, 1.10),
                textcoords=("data", "axes fraction"), ha="center", fontsize=8,
                color=S.INK_2, annotation_clip=False,
                arrowprops=dict(arrowstyle="-", color=S.INK_2, lw=0.7))
    for name, x, color, row in (("Youden J", thr["youden"], S.C4_YELLOW, 1.03),
                                ("F1-optimal", thr["f1_opt"], S.C3_MAGENTA, 1.03)):
        ax.axvline(x, color=color, lw=1, ls="--", alpha=0.85)
        ax.annotate(name, xy=(x, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(x, row), textcoords=("data", "axes fraction"),
                    ha="center", fontsize=7.5, color=S.INK, annotation_clip=False)
    cc = pol["policy_table_test"]["cost_cal"]["savings"] * 100
    ax.scatter([t_star], [cc], color=S.C1_BLUE, zorder=5, s=42,
               edgecolor="white", linewidth=1.2)
    ax.annotate("operating point", xy=(t_star, cc), xytext=(t_star + 0.09, cc + 3),
                fontsize=7.5, color=S.C1_BLUE,
                arrowprops=dict(arrowstyle="->", color=S.C1_BLUE, lw=0.8))
    ax.set_xlabel("decline threshold on probability")
    ax.set_ylabel(f"savings vs baseline (%), k = {k}")
    ax.set_xlim(0, 1)
    ax.legend(loc="lower right")
    ax.set_title("Savings vs decision threshold (test month)", pad=32)
    return S.savefig(fig, "figure_05_cost_vs_threshold")


# ---------------------------------------------------------------- fig 06 ----
def fig06_sensitivity_heatmap():
    pol = load_json("stage4_policies_test")
    grid = pol["sensitivity_test"]
    ks = list(grid.keys())
    policies = ["f1_opt", "youden", "cost_uncal", "cost_cal"]
    mat = np.array([[grid[k][p] for p in policies] for k in ks]) * 100

    from matplotlib.colors import TwoSlopeNorm
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    norm = TwoSlopeNorm(vcenter=0.0, vmin=min(mat.min(), -1.0),
                        vmax=max(mat.max(), 1.0))
    im = ax.imshow(mat, cmap=S.DIVERGING, norm=norm, aspect="auto")
    ax.set_xticks(range(len(policies)),
                  [POLICY_LABELS[p] for p in policies], fontsize=8)
    ax.set_yticks(range(len(ks)), [f"k = {k}" for k in ks])
    for i in range(len(ks)):
        for j in range(len(policies)):
            v = mat[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(norm(v) - 0.5) > 0.30 else S.INK)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="savings (%)", shrink=0.9)
    ax.set_title("Savings by false-decline cost multiplier k and policy (test month)")
    return S.savefig(fig, "figure_06_sensitivity_heatmap")


# ---------------------------------------------------------------- fig 07 ----
def fig07_decay_curves():
    drift = load_json("stage5_drift")
    metrics = [("tpr_at_5fpr", "TPR @ 5% FPR"), ("pr_auc", "PR-AUC"),
               ("savings", f"savings (k = {config.K_CENTRAL})")]
    colors = {"static": S.C1_BLUE, "expanding": S.C2_GREEN,
              "sliding": S.C3_MAGENTA}

    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.5), sharex=True)
    styles = {"static": dict(ls="-", lw=2.6, alpha=0.9),
              "expanding": dict(ls="--", lw=1.8),
              "sliding": dict(ls="-", lw=1.8)}
    for ax, (mkey, mlabel) in zip(axes, metrics):
        for pol, rows in drift["weekly"].items():
            wk = [r["week"] for r in rows]
            v = [r[mkey] for r in rows]
            ax.plot(wk, v, color=colors[pol], label=pol, marker="o",
                    markersize=3.2, **styles[pol])
        month5_start = min(r["week"] for r in drift["weekly"]["static"]
                           if r["month"] == config.TEST_MONTH)
        ax.axvline(month5_start - 0.5, color=S.BASELINE, lw=1, ls=":")
        ax.set_title(mlabel)
        ax.set_xlabel("week index")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.09))
    fig.suptitle("Weekly out-of-time decay by retraining policy (months 4-5)",
                 fontsize=10.5, fontweight="semibold", y=1.02)
    fig.text(0.5, 0.955, "static and expanding coincide until the first retrain "
             "(dotted line) - expanding is dashed so both stay visible",
             ha="center", fontsize=8, color=S.INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return S.savefig(fig, "figure_07_decay_curves")


# ---------------------------------------------------------------- fig 08 ----
def fig08_drift_psi():
    psi = load_json("stage5_psi")
    adv = load_json("stage5_adversarial")

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5),
                             gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    weeks = psi["weeks"]
    top5 = sorted(psi["max_psi_per_feature"].items(),
                  key=lambda kv: -kv[1])[:5]
    for i, (feat, _) in enumerate(top5):
        ax.plot(weeks, psi["psi"][feat], color=S.CATEGORICAL[i], lw=1.4,
                label=feat)
    ax.plot(weeks, psi["mean_psi_per_week"], color=S.INK, lw=2.4,
            label="mean (top-20 features)")
    ax.axhline(0.2, color=S.BASELINE, ls="--", lw=1,
               label="PSI = 0.2 (major-shift rule of thumb)")
    ax.set_xlabel("week index")
    ax.set_ylabel("PSI vs training window")
    ax.set_title("Population stability of top-20 model features")
    # the mid-left band (PSI ~0.45-1.0) is empty of data across all weeks
    ax.legend(fontsize=6.8, loc="center left", bbox_to_anchor=(0.01, 0.47),
              frameon=True, facecolor=S.SURFACE, edgecolor="none",
              framealpha=0.95)

    ax = axes[1]
    runs = adv["internal_oot"]
    labels = ["all model\nfeatures", "excl. causal\naggregates"]
    aucs = [r["cv_auc_mean"] for r in runs]
    sds = [r["cv_auc_sd"] for r in runs]
    bars = ax.bar(labels, aucs, yerr=sds, width=0.5,
                  color=[S.C1_BLUE, S.C5_AQUA], capsize=3)
    ax.axhline(0.5, color=S.BASELINE, ls="--", lw=1)
    ax.text(0.5, 0.515, "0.5 = indistinguishable", fontsize=7, color=S.INK_2,
            ha="center",
            bbox=dict(facecolor=S.SURFACE, edgecolor="none", pad=1.2))
    for b, v in zip(bars, aucs):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}",
                ha="center", fontsize=8, color=S.INK)
    ax.set_ylim(0.45, 1.02)
    ax.set_ylabel("train-vs-OOT classifier AUC")
    ax.set_title("Adversarial validation")
    return S.savefig(fig, "figure_08_psi_adversarial")


# ---------------------------------------------------------------- fig 09 ----
def fig09_ablation():
    abl = load_json("stage6_ablation")
    res = abl["results"]
    fwd = sorted([n for n in res if n.startswith("fwd_")],
                 key=lambda n: res[n]["n_features"])
    lobo = [n for n in res if n.startswith("lobo_")]

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    ax = axes[0]
    x = np.arange(len(fwd))
    tpr = [res[n]["tpr_at_5fpr"] for n in fwd]
    sav = [res[n]["savings"] for n in fwd]
    ax.plot(x, tpr, color=S.C1_BLUE, marker="o", label="TPR @ 5% FPR")
    ax.plot(x, sav, color=S.C2_GREEN, marker="s", label="savings (k=0.15)")
    ax.set_xticks(x, [n.replace("fwd_", "") for n in fwd], fontsize=7.2)
    ax.set_xlabel("cumulative feature blocks")
    ax.set_ylabel("validation score")
    ax.set_title("Forward addition")
    ax.legend()

    ax = axes[1]
    labels = [n.replace("lobo_minus_", "w/o ") for n in lobo]
    d_tpr = [res[n]["delta_tpr_at_5fpr"] * 100 for n in lobo]
    d_sav = [res[n]["delta_savings"] * 100 for n in lobo]
    yy = np.arange(len(lobo))
    ax.barh(yy + 0.19, d_tpr, height=0.36, color=S.C1_BLUE,
            label="Δ TPR@5%FPR (pp)")
    ax.barh(yy - 0.19, d_sav, height=0.36, color=S.C2_GREEN,
            label="Δ savings (pp)")
    ax.axvline(0, color=S.BASELINE, lw=1)
    ax.set_yticks(yy, labels, fontsize=8)
    ax.set_xlabel("change vs full configuration (validation, pp)")
    ax.set_title("Leave-one-block-out")
    ax.legend(fontsize=7.2)
    fig.suptitle("Incremental validity: which data earns its complexity?",
                 fontsize=10.5, fontweight="semibold", y=1.03)
    return S.savefig(fig, "figure_09_ablation")


# ---------------------------------------------------------------- fig 10 ----
def fig10_shap():
    import shap

    npz = np.load(config.DATA_INTERIM / "shap_sample.npz", allow_pickle=True)
    values, base = npz["values"], float(npz["base_value"])
    names = list(npz["feature_names"])
    row_idx = npz["row_idx"]

    feats = pd.read_parquet(config.DATA_PROCESSED / "features.parquet",
                            columns=names)
    Xs = feats.iloc[row_idx].reset_index(drop=True)
    del feats
    scores = pd.read_parquet(config.DATA_PROCESSED / "scores.parquet")
    meta = scores.iloc[row_idx].reset_index(drop=True)
    thr = config.K_CENTRAL / (1 + config.K_CENTRAL)

    Xnum = Xs.copy()
    for c in Xnum.columns:
        if not pd.api.types.is_numeric_dtype(Xnum[c]):
            Xnum[c] = Xnum[c].astype("category").cat.codes.replace(-1, np.nan)
    ex = shap.Explanation(values=values, base_values=base,
                          data=Xnum.to_numpy(dtype="float32"),
                          feature_names=names)

    fig = plt.figure(figsize=(8.2, 5.2))
    shap.plots.beeswarm(ex, max_display=14, show=False)
    plt.title("TreeSHAP summary - final model, stratified test sample",
              fontsize=10.5, fontweight="semibold")
    out1 = S.savefig(plt.gcf(), "figure_10a_shap_beeswarm")

    declined = (meta["p_cal"] > thr).to_numpy()
    tp_pool = np.flatnonzero((meta["y"] == 1).to_numpy() & declined)
    fp_pool = np.flatnonzero((meta["y"] == 0).to_numpy() & declined)
    tp = tp_pool[np.argmax(meta["p_cal"].to_numpy()[tp_pool])]
    fp = fp_pool[np.argmax(meta["p_cal"].to_numpy()[fp_pool])]

    def display_row(i: int):
        """Raw feature values (real category strings, not codes) for waterfalls."""
        vals = []
        for c in names:
            v = Xs.iloc[i][c]
            if pd.isna(v):
                vals.append(np.nan)
            elif isinstance(v, float):
                vals.append(round(float(v), 3))
            else:
                vals.append(str(v))
        return np.array(vals, dtype=object)

    for idx, tag, title in ((tp, "tp", "True positive (fraud, declined)"),
                            (fp, "fp", "False positive (legitimate, declined)")):
        row_ex = shap.Explanation(values=values[int(idx)], base_values=base,
                                  data=display_row(int(idx)),
                                  feature_names=names)
        plt.figure(figsize=(7.2, 4.6))
        shap.plots.waterfall(row_ex, max_display=10, show=False)
        p = float(meta.loc[idx, "p_cal"])
        p_str = "1.000 (top calibration step)" if p >= 0.9995 else f"{p:.3f}"
        plt.title(f"{title} - TxnID {int(meta.loc[idx, 'TransactionID'])}, "
                  f"\\${meta.loc[idx, 'amt']:.2f}, calibrated p = {p_str}",
                  fontsize=9.5, fontweight="semibold")
        S.savefig(plt.gcf(), f"figure_10{'b' if tag == 'tp' else 'c'}_waterfall_{tag}")
    return out1


ALL_FIGS = [fig00_protocol,
            fig01_class_balance, fig02_volume_over_time, fig03_optimism_gap,
            fig04_reliability, fig05_cost_curves, fig06_sensitivity_heatmap,
            fig07_decay_curves, fig08_drift_psi, fig09_ablation, fig10_shap]


def main() -> None:
    S.apply_style()
    for fn in ALL_FIGS:
        try:
            path = fn()
            log.info("wrote %s", path)
        except FileNotFoundError as e:
            log.warning("skipping %s (missing artifact: %s)", fn.__name__, e)


if __name__ == "__main__":
    main()
