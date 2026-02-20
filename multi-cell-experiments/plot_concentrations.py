"""Plot atomic concentrations per cell from saved MC results.

Run after aupt_600k.py and hfzrtanb.py have completed.
Produces:
  results/aupt_600k_concentrations.png
  results/hfzrtanb_concentrations.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

RESULTS_DIR = Path(__file__).parent / "results"

# ─────────────────────────────────────────────────────────────────────────────
# AuPt  (2 cells, 2 elements)
# ─────────────────────────────────────────────────────────────────────────────
aupt = np.load(RESULTS_DIR / "aupt_600k.npz")
x_au = aupt["x_au"]          # shape (n_steps+1, 2)
mr   = aupt["molar_ratios"]   # shape (n_steps+1, 2)
n    = x_au.shape[0]
steps = np.arange(n)

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
fig.suptitle("AuPt at 600 K  —  Multi-cell MC (ORB v3)", fontsize=13)

cell_colors = ["tab:blue", "tab:orange"]
cell_labels = ["Cell 1 (FCC, Au-rich start)", "Cell 2 (FCC, Pt-rich start)"]

# ── left: Au fraction per cell ────────────────────────────────────────────────
ax = axes[0]
for i in range(2):
    ax.plot(steps, x_au[:, i], color=cell_colors[i], label=cell_labels[i], lw=1.2)
ax.axhline(0.5, color="gray", ls="--", lw=0.8, label="Target (50 %)")
ax.set_xlabel("MC step")
ax.set_ylabel("Au fraction  $x_{\\rm Au}$")
ax.set_title("Au atomic fraction per cell")
ax.set_ylim(-0.02, 1.02)
ax.legend(fontsize=8)

# ── centre: Pt fraction per cell ──────────────────────────────────────────────
ax = axes[1]
for i in range(2):
    ax.plot(steps, 1 - x_au[:, i], color=cell_colors[i], label=cell_labels[i], lw=1.2)
ax.axhline(0.5, color="gray", ls="--", lw=0.8, label="Target (50 %)")
ax.set_xlabel("MC step")
ax.set_ylabel("Pt fraction  $x_{\\rm Pt}$")
ax.set_title("Pt atomic fraction per cell")
ax.set_ylim(-0.02, 1.02)
ax.legend(fontsize=8)

# ── right: molar ratios ────────────────────────────────────────────────────────
ax = axes[2]
for i in range(2):
    ax.plot(steps, mr[:, i], color=cell_colors[i], label=cell_labels[i], lw=1.2)
ax.axhline(0.5, color="gray", ls="--", lw=0.8, label="Equal weight")
ax.set_xlabel("MC step")
ax.set_ylabel("Molar ratio  $x_i$")
ax.set_title("Lever-rule molar ratios")
ax.set_ylim(-0.02, 1.02)
ax.legend(fontsize=8)

fig.tight_layout()
out = RESULTS_DIR / "aupt_600k_concentrations.png"
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"Saved {out}")

# ── summary table ─────────────────────────────────────────────────────────────
print("\nAuPt final state:")
print(f"{'Cell':<6} {'x_Au':>8} {'x_Pt':>8} {'molar_ratio':>12}")
for i in range(2):
    print(f"  {i+1:<4} {x_au[-1,i]:>8.3f} {1-x_au[-1,i]:>8.3f} {mr[-1,i]:>12.5f}")
print(f"  Target:  x_Au = 0.500,  x_Pt = 0.500")


# ─────────────────────────────────────────────────────────────────────────────
# HfZrTaNb  (4 cells, 4 elements)
# ─────────────────────────────────────────────────────────────────────────────
hea = np.load(RESULTS_DIR / "hfzrtanb.npz")
comp = hea["comp_history"]    # shape (n_steps+1, 4, 4)
mr4  = hea["molar_ratios"]    # shape (n_steps+1, 4)
elements = hea["elements"]    # [72, 40, 73, 41]
n4       = comp.shape[0]
steps4   = np.arange(n4)

SYM = {72: "Hf", 40: "Zr", 73: "Ta", 41: "Nb"}
el_colors = {"Hf": "#1f77b4", "Zr": "#17becf", "Ta": "#d62728", "Nb": "#ff7f0e"}
cell_labels4 = [
    "Cell 1 (BCC, Nb-rich start)",
    "Cell 2 (BCC, Ta-rich start)",
    "Cell 3 (HCP, Hf-rich start)",
    "Cell 4 (HCP, Zr-rich start)",
]

# One panel per cell, showing all 4 elemental fractions
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True, sharex=True)
fig.suptitle("HfZrTaNb at 1 000 K  —  Multi-cell MC (ORB v3)", fontsize=13)

for ci, ax in enumerate(axes.flat):
    for ei, z in enumerate(elements):
        sym = SYM[int(z)]
        ax.plot(steps4, comp[:, ci, ei],
                color=el_colors[sym], label=sym, lw=1.2)
    ax.axhline(0.25, color="gray", ls="--", lw=0.8, label="Target (25 %)")
    ax.set_title(cell_labels4[ci], fontsize=9)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=8, loc="upper right")

for ax in axes[1]:
    ax.set_xlabel("MC step")
for ax in axes[:, 0]:
    ax.set_ylabel("Atomic fraction")

fig.tight_layout()
out4 = RESULTS_DIR / "hfzrtanb_concentrations.png"
fig.savefig(out4, dpi=150)
plt.close(fig)
print(f"\nSaved {out4}")

# ── also plot molar ratios for HfZrTaNb ───────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(9, 4))
cell_colors4 = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
for ci, lbl in enumerate(cell_labels4):
    ax2.plot(steps4, mr4[:, ci], color=cell_colors4[ci], label=lbl, lw=1.2)
ax2.axhline(0.25, color="gray", ls="--", lw=0.8, label="Equal weight")
ax2.set_xlabel("MC step")
ax2.set_ylabel("Molar ratio")
ax2.set_title("HfZrTaNb  —  lever-rule molar ratios")
ax2.set_ylim(-0.02, 1.02)
ax2.legend(fontsize=8)
fig2.tight_layout()
out4b = RESULTS_DIR / "hfzrtanb_molar_ratios.png"
fig2.savefig(out4b, dpi=150)
plt.close(fig2)
print(f"Saved {out4b}")

# ── summary table ─────────────────────────────────────────────────────────────
print("\nHfZrTaNb final state:")
header = f"{'Cell':<6}" + "".join(f"{SYM[int(z)]:>8}" for z in elements) + f"{'molar_ratio':>12}"
print(header)
for ci in range(4):
    row = f"  {ci+1:<4}"
    row += "".join(f"{comp[-1,ci,ei]:>8.3f}" for ei in range(4))
    row += f"  {mr4[-1,ci]:>10.5f}"
    print(row)
print(f"  Target:  Hf=0.25  Zr=0.25  Ta=0.25  Nb=0.25")
