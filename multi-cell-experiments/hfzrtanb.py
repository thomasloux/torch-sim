"""HfZrTaNb quaternary high-entropy alloy multi-cell MC.

Reproduces the HfZrTaNb simulation from:
  Niu, C., Rao, Y., Windl, W. et al.
  "Multi-cell Monte Carlo method for phase prediction."
  npj Comput Mater 5, 120 (2019).
  https://doi.org/10.1038/s41524-019-0259-z

Setup (following the paper):
  - 4 cells representing the two candidate phase types:
      Cell 1: BCC  — Nb-rich  (75 % Nb, 25 % Ta)
      Cell 2: BCC  — Ta-rich  (75 % Ta, 25 % Nb)
      Cell 3: HCP  — Hf-rich  (75 % Hf, 25 % Zr)
      Cell 4: HCP  — Zr-rich  (75 % Zr, 25 % Hf)
  - Overall target: equiatomic Hf₂₅Zr₂₅Ta₂₅Nb₂₅
  - Temperature: 1 000 K
  - 8 000 MC steps

Expected result (paper): the HEA separates into a BCC phase enriched in
Nb and Ta and an HCP phase enriched in Hf and Zr.  If the model and the
algorithm reproduce this, the BCC cells should shed Hf/Zr and the HCP
cells should shed Nb/Ta.

MLIP: ORB v3 (conservative, inf-cutoff, OMAT training set).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import torch
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import torch_sim as ts  # noqa: E402
from torch_sim.models.orb import OrbModel  # noqa: E402
from torch_sim.units import MetalUnits  # noqa: E402
from ase.io.trajectory import Trajectory  # noqa: E402
from sqs_utils import make_bcc_sqs, make_hcp_sqs  # noqa: E402

# ── configuration ─────────────────────────────────────────────────────────────
TEMPERATURE = 1_000.0        # K
N_STEPS     = 2_000
SEED        = 42
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── device ────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype  = torch.float32
print(f"Using device: {device}")

# ── load ORB v3 ───────────────────────────────────────────────────────────────
print("Loading ORB v3 conservative model …")
from orb_models.forcefield import pretrained  # noqa: E402

orb_ff = pretrained.orb_v3_conservative_inf_omat(device=device, precision="float32-high")
model  = OrbModel(model=orb_ff, device=device, dtype=dtype)
print("Model loaded.")

# ── atomic numbers ────────────────────────────────────────────────────────────
Z_HF, Z_ZR, Z_TA, Z_NB = 72, 40, 73, 41
ALL_Z = [Z_HF, Z_ZR, Z_TA, Z_NB]
SYM   = {Z_HF: "Hf", Z_ZR: "Zr", Z_TA: "Ta", Z_NB: "Nb"}

# ── lattice parameters (experimental, Å) ──────────────────────────────────────
# BCC: Ta and Nb are isostructural (a ≈ 3.30 Å)
a_ta_bcc = 3.30
a_nb_bcc = 3.30

# HCP: Hf (a=3.19, c=5.05) and Zr (a=3.23, c=5.15)
a_hf, c_hf = 3.19, 5.05
a_zr, c_zr = 3.23, 5.15

print("Generating SQS structures …")

# # Cell 1: BCC Nb-rich SQS  (75 % Nb, 25 % Ta) → 24 Nb + 8 Ta in 32-atom BCC
# a_bcc1 = 0.75 * a_nb_bcc + 0.25 * a_ta_bcc
# cell1_ase = make_bcc_sqs(["Nb", "Ta"], a_bcc1,
#                           {"Nb": 0.75, "Ta": 0.25},
#                           repeat=(2, 2, 4), random_seed=SEED)

# # Cell 2: BCC Ta-rich SQS  (75 % Ta, 25 % Nb) → 24 Ta + 8 Nb in 32-atom BCC
# a_bcc2 = 0.75 * a_ta_bcc + 0.25 * a_nb_bcc
# cell2_ase = make_bcc_sqs(["Nb", "Ta"], a_bcc2,
#                           {"Ta": 0.75, "Nb": 0.25},
#                           repeat=(2, 2, 4), random_seed=SEED + 1)

# # Cell 3: HCP Hf-rich SQS  (75 % Hf, 25 % Zr) → 27 Hf + 9 Zr in 36-atom HCP
# a_hcp3 = 0.75 * a_hf + 0.25 * a_zr
# c_hcp3 = 0.75 * c_hf + 0.25 * c_zr
# cell3_ase = make_hcp_sqs(["Hf", "Zr"], a_hcp3, c_hcp3,
#                           {"Hf": 0.75, "Zr": 0.25},
#                           repeat=(3, 3, 2), random_seed=SEED + 2)

# # Cell 4: HCP Zr-rich SQS  (75 % Zr, 25 % Hf) → 27 Zr + 9 Hf in 36-atom HCP
# a_hcp4 = 0.25 * a_hf + 0.75 * a_zr
# c_hcp4 = 0.25 * c_hf + 0.75 * c_zr
# cell4_ase = make_hcp_sqs(["Hf", "Zr"], a_hcp4, c_hcp4,
#                           {"Hf": 0.25, "Zr": 0.75},
#                           repeat=(3, 3, 2), random_seed=SEED + 3)

# Cell 1: BCC Nb-rich SQS  (75 % Nb, 25 % Ta) → 24 Nb + 8 Ta in 32-atom BCC
a_bcc1 = 0.75 * a_nb_bcc + 0.25 * a_ta_bcc
cell1_ase = make_bcc_sqs(["Nb", "Ta"], a_bcc1,
                          {"Nb": 0.75, "Ta": 0.25},
                          repeat=(4, 4, 8), random_seed=SEED)

# Cell 2: BCC Ta-rich SQS  (75 % Ta, 25 % Nb) → 24 Ta + 8 Nb in 32-atom BCC
a_bcc2 = 0.75 * a_ta_bcc + 0.25 * a_nb_bcc
cell2_ase = make_bcc_sqs(["Nb", "Ta"], a_bcc2,
                          {"Ta": 0.75, "Nb": 0.25},
                          repeat=(4, 4, 8), random_seed=SEED + 1)

# Cell 3: HCP Hf-rich SQS  (75 % Hf, 25 % Zr) → 27 Hf + 9 Zr in 36-atom HCP
a_hcp3 = 0.75 * a_hf + 0.25 * a_zr
c_hcp3 = 0.75 * c_hf + 0.25 * c_zr
cell3_ase = make_hcp_sqs(["Hf", "Zr"], a_hcp3, c_hcp3,
                          {"Hf": 0.75, "Zr": 0.25},
                          repeat=(6, 6, 4), random_seed=SEED + 2)

# Cell 4: HCP Zr-rich SQS  (75 % Zr, 25 % Hf) → 27 Zr + 9 Hf in 36-atom HCP
a_hcp4 = 0.25 * a_hf + 0.75 * a_zr
c_hcp4 = 0.25 * c_hf + 0.75 * c_zr
cell4_ase = make_hcp_sqs(["Hf", "Zr"], a_hcp4, c_hcp4,
                          {"Hf": 0.25, "Zr": 0.75},
                          repeat=(6, 6, 4), random_seed=SEED + 3)

cells_ase = [cell1_ase, cell2_ase, cell3_ase, cell4_ase]
labels    = ["BCC Nb-rich", "BCC Ta-rich", "HCP Hf-rich", "HCP Zr-rich"]

for i, (atoms, lbl) in enumerate(zip(cells_ase, labels), 1):
    counts = {SYM[z]: int((atoms.numbers == z).sum()) for z in ALL_Z
              if (atoms.numbers == z).sum() > 0}
    print(f"Cell {i} ({lbl}, {len(atoms)} atoms): {counts}")

# Convert to SimState
cells_sim = [ts.initialize_state(a, device=device, dtype=dtype) for a in cells_ase]

# Target: equiatomic  Hf:Zr:Ta:Nb = 1:1:1:1
target_composition = {Z_HF: 1.0, Z_ZR: 1.0, Z_TA: 1.0, Z_NB: 1.0}

# ── optional: quick FIRE relaxation ───────────────────────────────────────────
print("\nRelaxing initial cells with FIRE …")
relaxed = []
for i, cell in enumerate(cells_sim):
    cell_relaxed = ts.optimize(cell, model, optimizer=ts.Optimizer.fire,
                               convergence_fn=ts.generate_force_convergence_fn(0.05),
                               max_steps=200, pbar=False)
    relaxed.append(cell_relaxed)
    print(f"  Cell {i+1} relaxed.")
cells_sim = relaxed
print("Relaxation done.")

# ── multi-cell MC ─────────────────────────────────────────────────────────────
print(f"\nRunning multi-cell MC @ {TEMPERATURE} K  ({N_STEPS} steps) …")

kT  = TEMPERATURE * MetalUnits.temperature
rng = torch.Generator(device=device)
rng.manual_seed(SEED)

state = ts.multi_cell_mc_init(
    cells_sim, model, target_composition=target_composition
)

# ── ASE trajectory writers ────────────────────────────────────────────────────
trajs = [
    Trajectory(str(RESULTS_DIR / f"hfzrtanb_cell{i + 1}.traj"), "w")
    for i in range(state.n_cells)
]
for i, cell in enumerate(state.cells):
    trajs[i].write(cell.to_atoms()[0])

n_cells = state.n_cells
molar_ratio_history: list[np.ndarray] = [state.molar_ratios.cpu().numpy()]
energy_history:      list[float]      = [state.total_energy.item()]


def composition_matrix(s: ts.MultiCellMCState) -> np.ndarray:
    """Return (n_cells × n_elements) fraction matrix."""
    mat = np.zeros((n_cells, len(ALL_Z)))
    for ci, cell in enumerate(s.cells):
        n_total = cell.n_atoms
        for ej, z in enumerate(ALL_Z):
            mat[ci, ej] = float((cell.atomic_numbers == z).sum()) / n_total
    return mat


comp_history: list[np.ndarray] = [composition_matrix(state)]

for _step in tqdm(range(N_STEPS), desc="MC steps"):
    state = ts.multi_cell_mc_step(state, model, kT=kT, rng=rng)
    for i, cell in enumerate(state.cells):
        trajs[i].write(cell.to_atoms()[0])
    molar_ratio_history.append(state.molar_ratios.cpu().numpy())
    energy_history.append(state.total_energy.item())
    comp_history.append(composition_matrix(state))

for traj in trajs:
    traj.close()

# ── analysis ──────────────────────────────────────────────────────────────────
phases       = ts.analyze_phases(state, threshold=0.01)
molar_ratios = phases["molar_ratios"].cpu().numpy()
compositions = phases["compositions"]

print("\n" + "=" * 60)
print("RESULTS: HfZrTaNb at 1 000 K")
print("=" * 60)
print(f"Steps completed   : {state.step_counter}")
print(f"Molar ratios      : {np.round(molar_ratios, 4)}")
print(f"Total energy      : {phases['total_energy']:.4f} eV")
print(f"Acceptance rates  : {phases['acceptance_rates'].cpu().numpy()}")
print(f"Stable phases     : cells {phases['stable_phases']}")

for i, (comp, ratio, lbl) in enumerate(zip(compositions, molar_ratios, labels)):
    n_total = sum(comp.values())
    fracs   = {SYM[z]: comp.get(z, 0) / n_total for z in ALL_Z} if n_total else {}
    print(f"\n  Cell {i+1} ({lbl})  molar ratio = {ratio:.4f}:")
    for sym, frac in fracs.items():
        print(f"    {sym}: {frac:.3f}")

# Expected: BCC cells (1 & 2) enriched in Nb/Ta; HCP cells (3 & 4) in Hf/Zr
bcc_nb_ta = np.array([compositions[j].get(Z_NB, 0) + compositions[j].get(Z_TA, 0)
                      for j in [0, 1]])
hcp_hf_zr = np.array([compositions[j].get(Z_HF, 0) + compositions[j].get(Z_ZR, 0)
                      for j in [2, 3]])
bcc_total = np.array([sum(compositions[j].values()) for j in [0, 1]])
hcp_total = np.array([sum(compositions[j].values()) for j in [2, 3]])
frac_nb_ta_bcc = (bcc_nb_ta / np.where(bcc_total > 0, bcc_total, 1)).mean()
frac_hf_zr_hcp = (hcp_hf_zr / np.where(hcp_total > 0, hcp_total, 1)).mean()
print(f"\nPhase-separation check:")
print(f"  BCC cells:  mean Nb+Ta fraction = {frac_nb_ta_bcc:.3f}  "
      f"(expected > 0.5 for BCC phase)")
print(f"  HCP cells:  mean Hf+Zr fraction = {frac_hf_zr_hcp:.3f}  "
      f"(expected > 0.5 for HCP phase)")

# ── save data ─────────────────────────────────────────────────────────────────
np.savez(
    RESULTS_DIR / "hfzrtanb.npz",
    molar_ratios = np.array(molar_ratio_history),
    energies     = np.array(energy_history),
    comp_history = np.array(comp_history),
    final_ratios = molar_ratios,
    elements     = np.array(ALL_Z),
)
print(f"\nData saved to {RESULTS_DIR / 'hfzrtanb.npz'}")
print(f"Trajectories saved to {RESULTS_DIR}/hfzrtanb_cell{{1..4}}.traj")

# ── plot ──────────────────────────────────────────────────────────────────────
steps    = np.arange(len(molar_ratio_history))
mr_arr   = np.array(molar_ratio_history)
comp_arr = np.array(comp_history)   # shape: (n_steps+1, 4 cells, 4 elements)

COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

# Molar ratios
for i, lbl in enumerate(labels):
    axes[0].plot(steps, mr_arr[:, i], color=COLORS[i], label=lbl)
axes[0].set_ylabel("Molar ratio")
axes[0].set_title("HfZrTaNb at 1 000 K  —  Multi-cell MC (ORB v3)")
axes[0].legend(fontsize=8)
axes[0].set_ylim(0, 1)
axes[0].axhline(0.25, color="gray", ls="--", lw=0.8)

# Composition (fraction of each element per cell)
elem_colors = {"Hf": "tab:purple", "Zr": "tab:cyan",
               "Ta": "tab:brown",  "Nb": "tab:pink"}
ei_map = {z: i for i, z in enumerate(ALL_Z)}
for ci, lbl in enumerate(labels):
    for z in ALL_Z:
        ei = ei_map[z]
        sym = SYM[z]
        ls = "-" if ci < 2 else "--"
        axes[1].plot(steps, comp_arr[:, ci, ei],
                     color=elem_colors[sym], ls=ls, lw=0.8,
                     label=f"{sym} (cell {ci+1})" if ci == 0 or ci == 2 else None)
axes[1].set_ylabel("Elemental fraction per cell")
axes[1].legend(fontsize=7, ncol=2)
axes[1].axhline(0.25, color="gray", ls="--", lw=0.8)

axes[2].plot(steps, energy_history, color="tab:green", lw=0.8)
axes[2].set_xlabel("MC step")
axes[2].set_ylabel("Weighted energy (eV)")

fig.tight_layout()
fig.savefig(RESULTS_DIR / "hfzrtanb.png", dpi=150)
print(f"Plot saved to {RESULTS_DIR / 'hfzrtanb.png'}")
plt.close(fig)
