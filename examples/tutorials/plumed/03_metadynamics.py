# %%
# /// script
# dependencies = [
#     "torch_sim_atomistic[plumed, io]"
# ]
# ///


# %% [markdown]
"""
# Well-Tempered Metadynamics

**Metadynamics** accelerates sampling by periodically depositing Gaussian "hills"
along a collective variable (CV), discouraging the system from revisiting
previously explored configurations. Over time, the deposited bias fills the
free-energy basins, allowing transitions between metastable states.

**Well-tempered metadynamics (WTMetaD)** uses a decreasing hill height so the
bias converges to a fraction of the true free-energy surface:

$$V_{\\text{bias}}(\\xi, t) = \\sum_{t' < t} W(t') \\exp\\!\\left(-\\frac{[\\xi - \\xi(t')]^2}{2\\sigma^2}\\right)$$

where $W(t') = W_0 \\exp\\!\\left(-\\frac{V_{\\text{bias}}(\\xi(t'), t')}{k_B \\Delta T}\\right)$
and $\\Delta T = (\\gamma - 1) T$ with $\\gamma$ the *bias factor*.

In TorchSim, this is just another PLUMED input — swap `PRINT` for `METAD`.

This tutorial covers:
1. Setting up `METAD` in the PLUMED input.
2. Choosing METAD parameters (SIGMA, HEIGHT, PACE, BIASFACTOR).
3. Running well-tempered metadynamics.
4. Reading the HILLS and COLVAR files.
5. Estimating the free-energy surface from the accumulated bias.
"""

# %%
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from ase.build import bulk

import torch_sim as ts
from torch_sim.models.lennard_jones import LennardJonesModel
from torch_sim.plumed import PlumedModel

torch.manual_seed(7)

# %% [markdown]
"""
## 1. System Setup

We use Ar FCC again. Metadynamics on a toy LJ system is pedagogical:
the 1D free-energy profile along a nearest-neighbour distance can be
compared to the analytical LJ pair potential.
"""

# %%
SIGMA_LJ = 3.405   # Å
EPSILON   = 0.0104  # eV
CUTOFF    = 2.5 * SIGMA_LJ

DEVICE = torch.device("cpu")
DTYPE  = torch.float32

lj_model = LennardJonesModel(
    sigma=SIGMA_LJ,
    epsilon=EPSILON,
    cutoff=CUTOFF,
    device=DEVICE,
    dtype=DTYPE,
    compute_forces=True,
    compute_stress=False,
    use_neighbor_list=True,
)

ar_atoms  = bulk("Ar", "fcc", a=5.26, cubic=True).repeat([2, 2, 2])

TIMESTEP    = 0.001   # ps
TEMPERATURE = 100     # K
kT          = TEMPERATURE * 8.617333e-5   # eV ≈ 0.00862 eV at 100 K

print(f"kT = {kT*1000:.2f} meV = {kT*96.485:.3f} kJ/mol")

# %% [markdown]
"""
## 2. Choosing METAD Parameters

| Parameter    | Meaning                               | Typical choice                          |
|--------------|---------------------------------------|-----------------------------------------|
| `SIGMA`      | Gaussian width (nm)                   | ~½ × typical CV fluctuation             |
| `HEIGHT`     | Initial Gaussian height (kJ/mol)      | A few kJ/mol; should be ≪ barrier       |
| `PACE`       | Steps between hill deposits           | 50–500; shorter → faster fill           |
| `BIASFACTOR` | Ratio $(T + \\Delta T)/T$, well-tempered | 5–20; higher → slower convergence, wider exploration |

For our Ar system at 100 K, thermal fluctuations of `d12` are ~0.02–0.05 nm,
so `SIGMA=0.02` is appropriate. The LJ well depth is 0.0104 eV = 1.0 kJ/mol,
so `HEIGHT=0.1 kJ/mol` per hill is gentle.
"""

# %%
work_dir = Path("03_metadynamics/")
# Create the directory if it doesn't exist
work_dir.mkdir(parents=True, exist_ok=True)

HILLS_FILE  = str(work_dir / "HILLS")
COLVAR_FILE = str(work_dir / "COLVAR")

# METAD parameters (all in PLUMED's native units: nm, kJ/mol)
SIGMA_NM      = 0.02    # Gaussian width (nm) — ~½ × thermal fluctuation
HEIGHT_KJMOL  = 0.1     # Initial hill height (kJ/mol)
PACE          = 25      # Deposit a hill every 25 steps
BIASFACTOR    = 8       # Well-tempered bias factor γ

plumed_metad = PlumedModel(
    model=lj_model,
    plumed_input=[
        # Collective variable: distance between atoms 1 and 2
        "d12: DISTANCE ATOMS=1,2",
        # Well-tempered metadynamics
        (
            f"METAD LABEL=metad ARG=d12 "
            f"SIGMA={SIGMA_NM} "
            f"HEIGHT={HEIGHT_KJMOL} "
            f"PACE={PACE} "
            f"BIASFACTOR={BIASFACTOR} "
            f"FILE={HILLS_FILE}"
        ),
        # Output: CV value and metadynamics bias
        f"PRINT ARG=d12,metad.bias STRIDE=5 FILE={COLVAR_FILE}",
    ],
    timestep=TIMESTEP,
    kT=kT,
)

print("PlumedModel with METAD created.")
print(f"HILLS → {HILLS_FILE}")
print(f"COLVAR → {COLVAR_FILE}")

# %% [markdown]
"""
## 3. Running Well-Tempered Metadynamics

We run 2 000 steps (~2 ps). In practice, metadynamics requires much longer
runs (hundreds of ns), but the qualitative behaviour — increasing bias,
growing exploration range — is visible even over short runs.
"""

# %%
N_STEPS = 20000

final_state = ts.integrate(
    system=ar_atoms,
    model=plumed_metad,
    integrator=ts.Integrator.nvt_langevin,
    n_steps=N_STEPS,
    temperature=TEMPERATURE,
    timestep=TIMESTEP,
)

n_hills = len(list(open(HILLS_FILE))) - 3  # subtract 3 header lines
print(f"Simulation done. Steps: {plumed_metad.step}")
print(f"Hills deposited: {n_hills}")

del plumed_metad  # free PLUMED resources before analysis

# %% [markdown]
"""
## 4. Reading the HILLS File

Each row of the HILLS file records one deposited Gaussian:
`time | CV_centre | sigma | height | bias_factor`

With well-tempered metadynamics the heights decrease over time as the bias
fills the free-energy surface.
"""

# %%
hills_data = np.loadtxt(HILLS_FILE, comments="#")
# Columns: time, centre (nm), sigma (nm), height (kJ/mol), biasfactor
time_ps   = hills_data[:, 0]
centre_nm = hills_data[:, 1]
height_kj = hills_data[:, 3]

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].plot(time_ps, centre_nm * 10, "o", ms=2, color="steelblue")
axes[0].set_xlabel("Time (ps)")
axes[0].set_ylabel("Hill centre (Å)")
axes[0].set_title("Metadynamics hill centres over time")

axes[1].plot(time_ps, height_kj, "o-", ms=2, color="coral")
axes[1].set_xlabel("Time (ps)")
axes[1].set_ylabel("Hill height (kJ/mol)")
axes[1].set_title("Well-tempered height decay")

fig.tight_layout()
plt.show()

# %% [markdown]
"""
## 5. Reading the COLVAR File

The COLVAR file records the CV and accumulated bias every `STRIDE` steps.
As the bias accumulates, the system is pushed towards higher free-energy regions,
increasing its exploration range.
"""

# %%
colvar_data = np.loadtxt(COLVAR_FILE, comments="#")
time_cv = colvar_data[:, 0]
d12_nm  = colvar_data[:, 1]
bias_kj = colvar_data[:, 2]

# Convert to more convenient units
d12_ang  = d12_nm * 10           # nm → Å
bias_eV  = bias_kj / 96.4853321  # kJ/mol → eV

fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)

axes[0].plot(time_cv, d12_ang, lw=0.6, color="steelblue")
axes[0].set_ylabel("d₁₂ (Å)")
axes[0].set_title("CV and bias during well-tempered metadynamics")

axes[1].plot(time_cv, bias_eV, lw=0.6, color="coral")
axes[1].set_ylabel("Accumulated bias (eV)")
axes[1].set_xlabel("Time (ps)")

fig.tight_layout()
plt.show()

# %% [markdown]
"""
## 6. Estimating the Free-Energy Surface

After a converged well-tempered metadynamics run, the accumulated bias is
proportional to the negative of the free-energy surface:

$$F(\\xi) \\approx -\\frac{\\gamma}{\\gamma - 1}\\, V_{\\text{bias}}(\\xi, t \\to \\infty)$$

We reconstruct $V_{\\text{bias}}$ on a grid by summing all deposited Gaussians.

> **Note**: 2 000 steps is far from converged. This shows the workflow; a real
> study needs orders of magnitude more sampling.
"""

# %%
# Reconstruct accumulated bias on a 1D grid
grid_ang = np.linspace(2.5, 6.5, 200)
grid_nm  = grid_ang / 10

# Sum all deposited hills
v_bias_grid_kj = np.zeros_like(grid_nm)
for row in hills_data:
    _, c_nm, s_nm, h_kj, _ = row
    v_bias_grid_kj += h_kj * np.exp(-0.5 * ((grid_nm - c_nm) / s_nm) ** 2)

# Well-tempered FES estimate: F ≈ -(γ/(γ-1)) * V_bias
fes_kj = -(BIASFACTOR / (BIASFACTOR - 1)) * v_bias_grid_kj
fes_kj -= fes_kj.min()  # shift minimum to zero

# Reference: analytical LJ pair potential (eV → kJ/mol)
r_ang = np.linspace(3.0, 6.5, 300)
lj_ref = 4 * EPSILON * ((SIGMA_LJ / r_ang) ** 12 - (SIGMA_LJ / r_ang) ** 6)
lj_ref_kj = lj_ref * 96.4853321
lj_ref_kj -= lj_ref_kj.min()

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(grid_ang, fes_kj, label="WTMetaD FES (2 000 steps)", color="steelblue")
ax.plot(r_ang, lj_ref_kj, "--", label="LJ pair potential (reference)", color="gray")
ax.set_xlim(3.0, 6.5)
ax.set_ylim(-0.1, 3.0)
ax.set_xlabel("d₁₂ (Å)")
ax.set_ylabel("Free energy (kJ/mol)")
ax.set_title("FES from well-tempered metadynamics")
ax.legend()
fig.tight_layout()
plt.show()

# %% [markdown]
"""
## 7. Tips for Production Metadynamics Runs

**SIGMA**
Choose ~½–1× the typical CV fluctuation in an unbiased short run.
Too small → too many hills, slow convergence.
Too large → poor resolution of the FES.

**HEIGHT and BIASFACTOR**
- Start with HEIGHT ≈ kBT (in kJ/mol) at the simulation temperature.
- BIASFACTOR 5–20 is typical. Higher values allow broader exploration but
  converge more slowly.

**PACE**
Every 50–500 steps is common. Faster deposition fills the surface quicker
but can cause instabilities if the hill size is large relative to the timestep.

**Convergence check**
- Plot the hill heights over time — they should decay smoothly.
- Compute the CV distribution: once it is flat, the FES is well converged.
- Recompute the FES at different time intervals and check for convergence.

**Restarting**
```python
plumed_model = PlumedModel(
    model=model,
    plumed_input=["d12: DISTANCE ATOMS=1,2",
                  "METAD ARG=d12 ... FILE=HILLS",
                  "PRINT ARG=d12 STRIDE=5 FILE=COLVAR"],
    timestep=0.001,
    kT=kT,
    restart=True,    # <-- tells PLUMED to append to existing HILLS/COLVAR
)
plumed_model.step = last_step  # sync the step counter
```
"""

# %% [markdown]
"""
## Summary

| PLUMED keyword | Purpose |
|----------------|---------|
| `METAD ARG=<cv>` | Apply metadynamics bias to CV |
| `SIGMA=<s>` | Gaussian width in PLUMED units (nm) |
| `HEIGHT=<h>` | Initial Gaussian height (kJ/mol) |
| `PACE=<n>` | Deposit a hill every `n` steps |
| `BIASFACTOR=<γ>` | Well-tempered parameter; set to `(T+ΔT)/T` |
| `FILE=HILLS` | Output file for deposited Gaussians |
| `metad.bias` | Accumulated bias (usable as a PRINT argument) |

For analysis beyond single-window histogram reweighting, see:
- [PLUMED nest](https://www.plumed-nest.org/) — curated enhanced-sampling workflows
- [PLUMED masterclass](https://www.plumed-nest.org/masterclass/) — video tutorials
- `plumed.read_as_pandas()` — load COLVAR/HILLS into a pandas DataFrame
"""
