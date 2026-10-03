"""Shared matplotlib style for report figures (static, light surface, print-oriented).

Colours are the validated reference categorical palette, assigned in fixed slot order and never cycled;
identity is always backed by a direct label or legend, never colour alone.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
SURFACE = "#ffffff"
SOURCE_COLOR = {"HCAST": SLOTS[0], "SWAA": SLOTS[1], "RE-Bench": SLOTS[2]}


def setup():
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.titlecolor": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "lines.linewidth": 2, "lines.markersize": 5, "legend.frameon": False,
    })
