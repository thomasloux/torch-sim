"""Hf-Zr binary alloy multi-cell MC.

Reproduces the Hf-Zr simulation from:
  Niu, C., Rao, Y., Windl, W. et al.
  "Multi-cell Monte Carlo method for phase prediction."
  npj Comput Mater 5, 120 (2019).
  https://doi.org/10.1038/s41524-019-0259-z

Setup (following the paper):
  - 2 HCP cells with complementary compositions
  - Cell 1: Hf-rich  (75 % Hf, 25 % Zr)
  - Cell 2: Zr-rich  (25 % Hf, 75 % Zr)
  - Overall target: equiatomic Hf₅₀Zr₅₀
  - Temperature: 700 K  (mid-range used in paper)
  - 5 000 MC steps

Expected result (paper): complete solid-solution miscibility — both cells
converge to near-equiatomic compositions and similar molar ratios (~0.5),
indicating no phase separation in the Hf-Zr system.

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
from sqs_utils import make_hcp_sqs  # noqa: E402

# ── configuration ─────────────────────────────────────────────────────────────
TEMPERATURE = 700.0          # K  (paper uses 400 / 700 / 1000 K)
N_STEPS     = 5_000
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
# Both cells are HCP (room-temperature structure for both Hf and Zr).
# SQS minimises Warren-Cowley pair correlations, matching the paper's approach.
a_hf, c_hf = 3.19, 5.05   # Å  (experimental)
a_zr, c_zr = 3.23, 5.15   # Å  (experimental)

# Atomic numbers
Z_HF, Z_ZR = 72, 40

print("Generating SQS structures …")

# Cell 1 — Hf-rich SQS: 27 Hf + 9 Zr  (75 / 25 %)
a1 = 0.75 * a_hf + 0.25 * a_zr
c1 = 0.75 * c_hf + 0.25 * c_zr
cell1_ase = make_hcp_sqs(["Hf", "Zr"], a1, c1,
                          {"Hf": 0.75, "Zr": 0.25},
                          repeat=(3, 3, 2), random_seed=SEED)

# Cell 2 — Zr-rich SQS: 9 Hf + 27 Zr  (25 / 75 %)
a2 = 0.25 * a_hf + 0.75 * a_zr
c2 = 0.25 * c_hf + 0.75 * c_zr
cell2_ase = make_hcp_sqs(["Hf", "Zr"], a2, c2,
                          {"Hf": 0.25, "Zr": 0.75},
                          repeat=(3, 3, 2), random_seed=SEED + 1)

for i, atoms in enumerate([cell1_ase, cell2_ase], 1):
    n_hf = (atoms.numbers == Z_HF).sum()
    n_zr = (atoms.numbers == Z_ZR).sum()
    print(f"Cell {i} ({len(atoms)} atoms): {n_hf} Hf + {n_zr} Zr  "
          f"[x_Hf={n_hf/len(atoms):.2f}]")

# Convert to SimState
cell1 = ts.initialize_state(cell1_ase, device=device, dtype=dtype)
cell2 = ts.initialize_state(cell2_ase, device=device, dtype=dtype)

target_composition = {Z_HF: 1.0, Z_ZR: 1.0}   # equiatomic

# ── optional: quick FIRE relaxation ───────────────────────────────────────────
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

kT  = TEMPERATURE * MetalUnits.temperature
rng = torch.Generator(device=device)
rng.manual_seed(SEED)

state = ts.multi_cell_mc_init(
    [cell1, cell2], model, target_composition=target_composition
)

# ── ASE trajectory writers ────────────────────────────────────────────────────
trajs = [
    Trajectory(str(RESULTS_DIR / "hfzr_cell1.traj"), "w"),
    Trajectory(str(RESULTS_DIR / "hfzr_cell2.traj"), "w"),
]
for i, cell in enumerate(state.cells):
    trajs[i].write(cell.to_atoms()[0])

molar_ratio_history: list[np.ndarray] = [state.molar_ratios.cpu().numpy()]
energy_history:      list[float]      = [state.total_energy.item()]

def hf_fractions(s: ts.MultiCellMCState) -> np.ndarray:
    return np.array([
        (cell.atomic_numbers == Z_HF).float().mean().item()
        for cell in s.cells
    ])

x_hf_history: list[np.ndarray] = [hf_fractions(state)]

for _step in tqdm(range(N_STEPS), desc="MC steps"):
    state = ts.multi_cell_mc_step(state, model, kT=kT, rng=rng)
    for i, cell in enumerate(state.cells):
        trajs[i].write(cell.to_atoms()[0])
    molar_ratio_history.append(state.molar_ratios.cpu().numpy())
    energy_history.append(state.total_energy.item())
    x_hf_history.append(hf_fractions(state))

for traj in trajs:
    traj.close()

# ── analysis ──────────────────────────────────────────────────────────────────
phases       = ts.analyze_phases(state, threshold=0.01)
molar_ratios = phases["molar_ratios"].cpu().numpy()
compositions = phases["compositions"]

print("\n" + "=" * 60)
print("RESULTS: Hf-Zr at 700 K")
print("=" * 60)
print(f"Steps completed   : {state.step_counter}")
print(f"Molar ratios      : {molar_ratios}")
print(f"Total energy      : {phases['total_energy']:.4f} eV")
print(f"Acceptance rates  : {phases['acceptance_rates'].cpu().numpy()}")
print(f"Stable phases     : cells {phases['stable_phases']}")

# Check convergence toward equiatomic compositions (full miscibility signal)
final_x_hf = x_hf_history[-1]
delta = abs(final_x_hf[0] - final_x_hf[1])
if delta < 0.05:
    print("\n→ Cells have converged to similar Hf fractions: FULL MISCIBILITY")
else:
    print(f"\n→ Hf fraction difference between cells: Δx_Hf = {delta:.3f}  (PHASE SEPARATION?)")

for i, (comp, ratio) in enumerate(zip(compositions, molar_ratios)):
    n_hf    = comp.get(Z_HF, 0)
    n_zr    = comp.get(Z_ZR, 0)
    n_total = n_hf + n_zr
    x_hf    = n_hf / n_total if n_total else 0.0
    print(f"\n  Cell {i + 1}  (molar ratio = {ratio:.4f}):")
    print(f"    {n_hf} Hf + {n_zr} Zr  →  x_Hf = {x_hf:.3f},  x_Zr = {1 - x_hf:.3f}")

# ── save data ─────────────────────────────────────────────────────────────────
np.savez(
    RESULTS_DIR / "hfzr.npz",
    molar_ratios = np.array(molar_ratio_history),
    energies     = np.array(energy_history),
    x_hf         = np.array(x_hf_history),
    final_ratios = molar_ratios,
)
print(f"\nData saved to {RESULTS_DIR / 'hfzr.npz'}")
print(f"Trajectories saved to {RESULTS_DIR}/hfzr_cell1.traj and hfzr_cell2.traj")

# ── plot ──────────────────────────────────────────────────────────────────────
steps    = np.arange(len(molar_ratio_history))
x_hf_arr = np.array(x_hf_history)

fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)

for i in range(2):
    axes[0].plot(steps, np.array(molar_ratio_history)[:, i], label=f"Cell {i+1}")
axes[0].set_ylabel("Molar ratio")
axes[0].set_title("Hf-Zr at 700 K  —  Multi-cell MC (ORB v3)")
axes[0].legend()
axes[0].set_ylim(0, 1)
axes[0].axhline(0.5, color="gray", ls="--", lw=0.8)

for i in range(2):
    axes[1].plot(steps, x_hf_arr[:, i], label=f"Cell {i+1}")
axes[1].set_ylabel("Hf fraction $x_{\\rm Hf}$")
axes[1].legend()
axes[1].axhline(0.5, color="gray", ls="--", lw=0.8)

axes[2].plot(steps, energy_history, color="tab:green", lw=0.8)
axes[2].set_xlabel("MC step")
axes[2].set_ylabel("Weighted energy (eV)")

fig.tight_layout()
fig.savefig(RESULTS_DIR / "hfzr.png", dpi=150)
print(f"Plot saved to {RESULTS_DIR / 'hfzr.png'}")
plt.close(fig)
