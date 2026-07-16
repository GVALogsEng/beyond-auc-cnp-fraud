"""Shared figure style: validated categorical palette, recessive chrome.

Palette follows a validated 8-slot categorical order (adjacent-pair CVD
Delta-E >= 8; assign slots in order, never cycle); sequential = one blue ramp;
diverging = blue <-> red with a neutral-gray midpoint. One axis per panel --
never dual axes.
"""
from __future__ import annotations

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

# categorical slots, fixed order (light mode)
C1_BLUE = "#2a78d6"
C2_GREEN = "#008300"
C3_MAGENTA = "#e87ba4"
C4_YELLOW = "#eda100"
C5_AQUA = "#1baf7a"
C6_ORANGE = "#eb6834"
CATEGORICAL = [C1_BLUE, C2_GREEN, C3_MAGENTA, C4_YELLOW, C5_AQUA, C6_ORANGE]

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

SEQ_BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf",
             "#184f95", "#0d366b"]
DIVERGING = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#0d366b", "#3987e5", "#cde2fb", "#f0efec",
                     "#f6c4c4", "#e34948", "#8f1f1f"])
SEQUENTIAL = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUES)


def apply_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 10.5,
        "axes.titleweight": "semibold",
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
    })


def savefig(fig, name: str) -> str:
    from src import config
    path = config.FIGURES / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return str(path)
