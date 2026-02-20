"""AuPt binary alloy multi-cell MC at 600 K.

Reproduces the AuPt simulation from:
  Niu, C., Rao, Y., Windl, W. et al.
  "Multi-cell Monte Carlo method for phase prediction."
  npj Comput Mater 5, 120 (2019).
  https://doi.org/10.1038/s41524-019-0259-z

Setup (following the paper):
  - 2 FCC cells with complementary compositions
  - Cell 1: Au-rich  (75 % Au, 25 % Pt)
  - Cell 2: Pt-rich  (25 % Au, 75 % Pt)
  - Overall target: equiatomic Au₅₀Pt₅₀
  - Temperature: 600 K
  - 5 000 MC steps

Expected result (paper): phase separation into a Pt-rich and an Au-rich FCC
phase at 600 K, consistent with the known Au-Pt miscibility gap.

MLIP: ORB v3 (conservative, inf-cutoff, OMAT training set).
"""

from __future__ import annotations

import warnings
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
from sqs_utils import make_fcc_sqs  # noqa: E402

# ── configuration ─────────────────────────────────────────────────────────────
TEMPERATURE = 600.0          # K
N_STEPS     = 1_000
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

# ── build initial cells (SQS) ─────────────────────────────────────────────────
# Both cells are FCC.  Lattice parameters via Vegard's law.
# SQS minimises Warren-Cowley pair correlations to mimic a random alloy,
# matching the approach of the original paper (Niu et al. 2019).
a_au = 4.078   # Å  (experimental)
a_pt = 3.923   # Å  (experimental)

# Atomic numbers
Z_AU, Z_PT = 79, 78

print("Generating SQS structures …")

supercell = (3, 3, 3)  # 32 atoms per cell
# Cell 1 — Au-rich SQS: 24 Au + 8 Pt  (75 / 25 %)
a1        = 0.75 * a_au + 0.25 * a_pt   # ≈ 4.039 Å
cell1_ase = make_fcc_sqs(["Au", "Pt"], a1,
                          {"Au": 0.75, "Pt": 0.25},
                          repeat=supercell, random_seed=SEED)

# Cell 2 — Pt-rich SQS: 8 Au + 24 Pt  (25 / 75 %)
a2        = 0.25 * a_au + 0.75 * a_pt   # ≈ 3.961 Å
cell2_ase = make_fcc_sqs(["Au", "Pt"], a2,
                          {"Au": 0.25, "Pt": 0.75},
                          repeat=supercell, random_seed=SEED + 1)

for i, atoms in enumerate([cell1_ase, cell2_ase], 1):
    n_au = (atoms.numbers == Z_AU).sum()
    n_pt = (atoms.numbers == Z_PT).sum()
    print(f"Cell {i} ({len(atoms)} atoms): {n_au} Au + {n_pt} Pt  "
          f"[x_Au={n_au/len(atoms):.2f}]")

# Convert to SimState
cell1 = ts.initialize_state(cell1_ase, device=device, dtype=dtype)
cell2 = ts.initialize_state(cell2_ase, device=device, dtype=dtype)

# Target: equiatomic  (total = 32 Au + 32 Pt across both cells)
target_composition = {Z_AU: 1.0, Z_PT: 1.0}   # ratios, not absolute counts

# ── optional: quick FIRE relaxation of each cell ──────────────────────────────
print("\nRelaxing initial cells with FIRE …")
cell1 = ts.optimize(cell1, model, optimizer=ts.Optimizer.fire,
                    convergence_fn=ts.generate_force_convergence_fn(0.05),
                    max_steps=200, pbar=False)
cell2 = ts.optimize(cell2, model, optimizer=ts.Optimizer.fire,
                    convergence_fn=ts.generate_force_convergence_fn(0.05),
                    max_steps=200, pbar=False)
print("Relaxation done.")

# ── multi-cell MC ─────────────────────────────────────────────────────────────
print(f"\nRunning multi-cell MC @ {TEMPERATURE} K  ({N_STEPS} steps) …")

kT  = TEMPERATURE * MetalUnits.temperature   # eV
rng = torch.Generator(device=device)
rng.manual_seed(SEED)

state = ts.multi_cell_mc_init(
    [cell1, cell2], model, target_composition=target_composition
)

# ── ASE trajectory writers ────────────────────────────────────────────────────
trajs = [
    Trajectory(str(RESULTS_DIR / "aupt_600k_cell1.traj"), "w"),
    Trajectory(str(RESULTS_DIR / "aupt_600k_cell2.traj"), "w"),
]
for i, cell in enumerate(state.cells):
    trajs[i].write(cell.to_atoms()[0])

molar_ratio_history: list[np.ndarray] = [state.molar_ratios.cpu().numpy()]
energy_history:      list[float]      = [state.total_energy.item()]

# fractions of Au in each cell over time
x_au_history: list[np.ndarray] = []

def au_fractions(s: ts.MultiCellMCState) -> np.ndarray:
    return np.array([
        (cell.atomic_numbers == Z_AU).float().mean().item()
        for cell in s.cells
    ])

x_au_history.append(au_fractions(state))

for _step in tqdm(range(N_STEPS), desc="MC steps"):
    state = ts.multi_cell_mc_step(state, model, kT=kT, rng=rng)
    for i, cell in enumerate(state.cells):
        trajs[i].write(cell.to_atoms()[0])
    molar_ratio_history.append(state.molar_ratios.cpu().numpy())
    energy_history.append(state.total_energy.item())
    x_au_history.append(au_fractions(state))

for traj in trajs:
    traj.close()

# ── analysis ──────────────────────────────────────────────────────────────────
phases = ts.analyze_phases(state, threshold=0.01)
molar_ratios = phases["molar_ratios"].cpu().numpy()
compositions = phases["compositions"]

print("\n" + "=" * 60)
print("RESULTS: AuPt at 600 K")
print("=" * 60)
print(f"Steps completed   : {state.step_counter}")
print(f"Molar ratios      : {molar_ratios}")
print(f"Total energy      : {phases['total_energy']:.4f} eV")
print(f"Acceptance rates  : {phases['acceptance_rates'].cpu().numpy()}")
print(f"Stable phases     : cells {phases['stable_phases']}")

for i, (comp, ratio) in enumerate(zip(compositions, molar_ratios)):
    n_au    = comp.get(Z_AU, 0)
    n_pt    = comp.get(Z_PT, 0)
    n_total = n_au + n_pt
    x_au    = n_au / n_total if n_total else 0.0
    print(f"\n  Cell {i + 1}  (molar ratio = {ratio:.4f}):")
    print(f"    {n_au} Au + {n_pt} Pt  →  x_Au = {x_au:.3f},  x_Pt = {1 - x_au:.3f}")

# ── save data ─────────────────────────────────────────────────────────────────
np.savez(
    RESULTS_DIR / "aupt_600k.npz",
    molar_ratios   = np.array(molar_ratio_history),
    energies       = np.array(energy_history),
    x_au           = np.array(x_au_history),
    final_ratios   = molar_ratios,
)
print(f"\nData saved to {RESULTS_DIR / 'aupt_600k.npz'}")
print(f"Trajectories saved to {RESULTS_DIR}/aupt_600k_cell1.traj and aupt_600k_cell2.traj")

# ── plot ──────────────────────────────────────────────────────────────────────
steps = np.arange(len(molar_ratio_history))
x_au_arr = np.array(x_au_history)

fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)

# Molar ratios
for i in range(2):
    axes[0].plot(steps, np.array(molar_ratio_history)[:, i], label=f"Cell {i+1}")
axes[0].set_ylabel("Molar ratio")
axes[0].set_title("AuPt at 600 K  —  Multi-cell MC (ORB v3)")
axes[0].legend()
axes[0].set_ylim(0, 1)
axes[0].axhline(0.5, color="gray", ls="--", lw=0.8)

# Au fractions per cell
for i in range(2):
    axes[1].plot(steps, x_au_arr[:, i], label=f"Cell {i+1}")
axes[1].set_ylabel("Au fraction $x_{\\rm Au}$")
axes[1].legend()
axes[1].axhline(0.5, color="gray", ls="--", lw=0.8)

# Total weighted energy
axes[2].plot(steps, energy_history, color="tab:green", lw=0.8)
axes[2].set_xlabel("MC step")
axes[2].set_ylabel("Weighted energy (eV)")

fig.tight_layout()
fig.savefig(RESULTS_DIR / "aupt_600k.png", dpi=150)
print(f"Plot saved to {RESULTS_DIR / 'aupt_600k.png'}")
plt.close(fig)
