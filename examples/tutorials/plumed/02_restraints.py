# %%
# /// script
# dependencies = [
#     "torch_sim_atomistic[plumed, io]"
# ]
# ///


# %% [markdown]
"""
# Umbrella Sampling with RESTRAINT

**Umbrella sampling** is one of the most widely used enhanced-sampling techniques.
The idea is simple: run several independent simulations, each biased by a harmonic
restraint centred at a different value of a collective variable (CV). This forces
the system to explore regions of configuration space that are rarely visited in
unbiased MD.

The harmonic bias added by `RESTRAINT` is:

$$V_{\\text{bias}}(\\xi) = \\frac{\\kappa}{2} (\\xi - \\xi_0)^2$$

where $\\xi$ is the CV, $\\xi_0$ is the restraint centre, and $\\kappa$ is the
force constant.

In this tutorial we:
1. Apply a `RESTRAINT` on an interatomic distance.
2. Verify that the bias modifies forces and energy.
3. Run umbrella windows along a distance coordinate.
4. Estimate the free-energy profile (PMF) with a simple histogram approach.
"""

# %%
# import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from ase.build import bulk

import torch_sim as ts
from torch_sim.models.lennard_jones import LennardJonesModel
from torch_sim.plumed import PlumedModel

torch.manual_seed(0)

# %% [markdown]
"""
## 1. System and Model

Same Ar FCC system as in `01_basics.py`.
"""

# %%
SIGMA = 3.405  # Å (Ar–Ar LJ diameter)
EPSILON = 0.0104  # eV
CUTOFF = 2.5 * SIGMA

DEVICE = torch.device("cpu")
DTYPE = torch.float32
lj_model = LennardJonesModel(
    sigma=SIGMA,
    epsilon=EPSILON,
    cutoff=CUTOFF,
    device=DEVICE,
    dtype=DTYPE,
    compute_forces=True,
    compute_stress=False,
)

ar_atoms = bulk("Ar", "fcc", a=5.26, cubic=True).repeat([2, 2, 2])
ar_state = ts.io.atoms_to_state(ar_atoms, DEVICE, DTYPE)

# LJ nearest-neighbour distance (pair minimum at r = 2^(1/6) * sigma)
r_min = 2 ** (1 / 6) * SIGMA  # ≈ 3.82 Å = 0.382 nm
print(f"LJ pair minimum: {r_min:.3f} Å = {r_min / 10:.4f} nm")

TIMESTEP = 0.001  # ps
TEMPERATURE = 100  # K
kT = TEMPERATURE * 8.617333e-5  # eV

# %% [markdown]
"""
## 2. How RESTRAINT Modifies Forces

Before running a full simulation, let's verify that a restraint at a different
distance from the actual nearest-neighbour distance produces a non-zero bias.

> **Unit note**: `AT` and `KAPPA` are in PLUMED's native units.
> - `AT=0.30` → restraint centre at 0.30 nm = 3.0 Å
> - `KAPPA=500` → force constant 500 kJ/mol/nm² ≈ 0.052 eV/Å²
"""

# %%
work_dir = Path("02_restraints/")

colvar_check = str(work_dir / "colvar_check.out")
plumed_check = PlumedModel(
    model=lj_model,
    plumed_input=[
        "d12: DISTANCE ATOMS=1,2",
        "RESTRAINT ARG=d12 AT=0.30 KAPPA=500.0",
        f"PRINT ARG=d12 STRIDE=1 FILE={colvar_check}",
    ],
    timestep=TIMESTEP,
    kT=kT,
)

# Compare unbiased vs biased output on the same state
unbiased_out = lj_model(ar_state)
biased_out = plumed_check(ar_state)

bias_energy = float(biased_out["energy"][0] - unbiased_out["energy"][0])
force_change = (biased_out["forces"] - unbiased_out["forces"]).norm(dim=1).max().item()

print(f"Bias energy on first step: {bias_energy:.6f} eV")
print(f"Max force change:          {force_change:.6f} eV/Å")

# %% [markdown]
"""
## 3. Running Umbrella Windows

We sweep the restraint centre `AT` from 3.0 Å (0.30 nm) to 5.0 Å (0.50 nm) in
five windows, running 300 steps each. Each window collects a COLVAR file with the
instantaneous distance.

In a real calculation you would:
- Use many more steps (tens of thousands) per window.
- Employ overlapping windows with enough spacing to ensure overlap of histograms.
- Apply WHAM or MBAR to recover the unbiased free-energy profile.
"""

# %%
# Restraint centres in nm (PLUMED native unit)
AT_CENTRES_NM = np.linspace(0.30, 0.50, 5)
KAPPA_NM = 10000.0  # kJ/mol/nm²
N_STEPS_PER_WINDOW = 2000

colvar_paths = []
histograms = []

for i, at_nm in enumerate(AT_CENTRES_NM):
    colvar_path = str(work_dir / f"COLVAR_w{i}.dat")
    colvar_paths.append(colvar_path)

    plumed_window = PlumedModel(
        model=lj_model,
        plumed_input=[
            "d12: DISTANCE ATOMS=1,2",
            f"RESTRAINT ARG=d12 AT={at_nm:.4f} KAPPA={KAPPA_NM:.1f}",
            f"PRINT ARG=d12 STRIDE=1 FILE={colvar_path}",
        ],
        timestep=TIMESTEP,
        kT=kT,
    )

    ts.integrate(
        system=ar_atoms,
        model=plumed_window,
        integrator=ts.Integrator.nvt_langevin,
        n_steps=N_STEPS_PER_WINDOW,
        temperature=TEMPERATURE,
        timestep=TIMESTEP,
    )
    print(f"Window {i + 1}/{len(AT_CENTRES_NM)}: AT={at_nm:.3f} nm — done")

    del plumed_window  # free PLUMED resources before next window
    # necessary to have PLUMED writing the COLVAR file #TODO solve this more elegantly

# %% [markdown]
"""
## 4. Loading and Plotting the Umbrella Windows

Each COLVAR file contains the time series of `d12` for its window. We can check
that the distributions are centred near their target distances.
"""

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# Left: raw time series
ax = axes[0]
for i, (path, at_nm) in enumerate(zip(colvar_paths, AT_CENTRES_NM)):
    data = np.loadtxt(path, comments="#")
    d12_ang = data[:, 1] * 10  # nm → Å
    ax.plot(data[:, 0], d12_ang, alpha=0.7, label=f"AT={at_nm:.2f} nm")
ax.set_xlabel("Time (ps)")
ax.set_ylabel("d₁₂ (Å)")
ax.set_title("Distance time series per window")
ax.legend(fontsize=7)

# Right: histograms
ax2 = axes[1]
bin_edges = np.linspace(2.5, 6.5, 40)
for i, (path, at_nm) in enumerate(zip(colvar_paths, AT_CENTRES_NM)):
    data = np.loadtxt(path, comments="#")
    d12_ang = data[:, 1] * 10
    counts, _ = np.histogram(d12_ang, bins=bin_edges, density=True)
    centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    ax2.plot(centres, counts, label=f"AT={at_nm:.2f} nm")
    ax2.axvline(at_nm * 10, ls="--", color=f"C{i}", alpha=0.5, lw=0.8)

ax2.set_xlabel("d₁₂ (Å)")
ax2.set_ylabel("Probability density")
ax2.set_title("Biased distance distributions")
ax2.legend(fontsize=7)

fig.tight_layout()
plt.show()

# %% [markdown]
"""
## 5. Simple Histogram-Based PMF Estimate

From each window, we compute the biased free energy $F_\\text{biased}(\\xi)$
and then remove the harmonic restraint to obtain the unbiased free energy
(this is the single-window histogram approach — full WHAM uses all windows
simultaneously, but the idea is the same).

$$F(\\xi) = -k_B T \\ln P_\\text{biased}(\\xi) - V_\\text{bias}(\\xi) + C$$

where the constant $C$ is chosen so that the profiles join smoothly.

> **Note**: With only 300 steps per window the statistics are very poor.
> This cell is purely illustrative of the workflow.
"""

# %%
kT_eV = kT  # eV
kappa_ev_ang2 = KAPPA_NM / (96.4853321 * 100)  # convert kJ/mol/nm² → eV/Å²

fig, ax = plt.subplots(figsize=(7, 4))
pmf_segments = []

for i, (path, at_nm) in enumerate(zip(colvar_paths, AT_CENTRES_NM)):
    at_ang = at_nm * 10  # nm → Å

    data = np.loadtxt(path, comments="#")
    d12_ang = data[:, 1] * 10
    counts, edges = np.histogram(d12_ang, bins=20, density=True)
    centres = 0.5 * (edges[:-1] + edges[1:])

    with np.errstate(divide="ignore"):
        f_biased = -kT_eV * np.log(np.where(counts > 0, counts, np.nan))

    v_bias = 0.5 * kappa_ev_ang2 * (centres - at_ang) ** 2
    pmf_raw = f_biased - v_bias

    # Shift so minimum is zero
    pmf_raw -= np.nanmin(pmf_raw)
    pmf_segments.append((centres, pmf_raw))

    ax.plot(centres, pmf_raw, "o-", ms=3, label=f"Window {i + 1}")

ax.set_xlabel("d₁₂ (Å)")
ax.set_ylabel("PMF (eV, shifted)")
ax.set_title("Per-window PMF estimate\n(full WHAM needed for converged result)")
ax.legend(fontsize=8)
fig.tight_layout()
plt.show()

# %% [markdown]
"""
## Summary

- `RESTRAINT ARG=<cv> AT=<centre_nm> KAPPA=<k_kJmol_nm2>` adds a harmonic bias
  to any PLUMED CV.
- `AT` and `KAPPA` are in **PLUMED's native units** (nm, kJ/mol, nm²).
- Run one `PlumedModel` per umbrella window.
- The COLVAR files contain the biased time series — feed them to WHAM or MBAR
  for an unbiased free-energy profile.

Next: [03_metadynamics.py](03_metadynamics.py) — automatically filling the free-energy
landscape with well-tempered metadynamics.
"""
