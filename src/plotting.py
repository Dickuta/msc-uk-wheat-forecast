"""
plotting.py

Shared matplotlib styling. Charts are rendered inline by the notebooks
(matplotlib inline backend) — no figure files are written to disk.

The palette is tuned to be readable in print (thesis) and on screen.
"""

import matplotlib

try:  # keep ``python script.py`` working outside a kernel
    matplotlib.use("module://matplotlib_inline.backend_inline")
except Exception:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

# Global style
plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "font.size": 10,
    }
)

# Color-blind-friendly qualitative palette.
PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
]
