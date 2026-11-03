"""Shared figure style for the VoI-Search manuscript (WSDM '27).

Categorical palette is colorblind-validated (CVD dE >= 8 adjacent pairs on a
white surface). Color assignment is fixed per method across every figure in
the paper; series identity is additionally carried by marker shape and line
style so figures survive grayscale printing.
"""
import matplotlib as mpl

BLUE = "#2a78d6"     # VoI-Search (ours)
ORANGE = "#eb6834"   # ReAct
AQUA = "#1baf7a"     # Self-RAG / entropy-threshold
YELLOW = "#eda100"   # Adaptive-RAG
MAGENTA = "#e87ba4"  # fixed-budget RAG
VIOLET = "#4a3aa7"   # RL cost-penalty
GRAY = "#6b6a66"     # oracle / reference lines


def apply_style():
    mpl.rcParams.update({
        "font.size": 7.5,
        "font.family": "sans-serif",
        "axes.titlesize": 8,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.4,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.30,
        "lines.linewidth": 1.4,
        "lines.markersize": 3.6,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
    })
