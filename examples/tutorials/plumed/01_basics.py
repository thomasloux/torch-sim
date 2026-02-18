# %%
# /// script
# dependencies = [
#     "torch_sim_atomistic[plumed, io]"
# ]
# ///


# %% [markdown]
"""
# PLUMED Basics: Collective Variables and the PlumedModel

[PLUMED](https://www.plumed.org) is the standard library for enhanced sampling in
molecular dynamics. It computes **collective variables** (CVs) — functions of atomic
positions that describe slow degrees of freedom — and optionally applies **bias forces**
to accelerate sampling.

`PlumedModel` wraps any TorchSim model, adding PLUMED bias on top of the unbiased
potential on every MD step. This tutorial covers:

1. Creating a `PlumedModel`
2. How collective variables are defined
3. Running a simulation and reading the `COLVAR` output file
4. The step counter and simulation restarts

We use a Lennard-Jones Argon system throughout — no external MLIP required.
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

# Reproducible results
torch.manual_seed(42)

# %% [markdown]
"""
## 1. System Setup

We start with a 2×2×2 FCC Argon supercell (32 atoms). The LJ parameters are the
standard Ar–Ar values from the literature.
"""

# %%
# Standard Lennard-Jones parameters for Argon
SIGMA = 3.405   # Å
EPSILON = 0.0104  # eV
CUTOFF = 2.5 * SIGMA  # Å — standard LJ cutoff

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
    use_neighbor_list=True,
)

# FCC Ar, 2×2×2 supercell (32 atoms)
ar_atoms = bulk("Ar", "fcc", a=5.26, cubic=True).repeat([2, 2, 2])
print(f"System: {len(ar_atoms)} Ar atoms")
print(f"Cell: {ar_atoms.cell.array.diagonal()} Å")
print(f"Nearest-neighbour distance: {5.26 / 2**0.5:.3f} Å = {5.26 / 2**0.5 / 10:.4f} nm")

# %% [markdown]
"""
## 2. Creating a PlumedModel

`PlumedModel` takes:
- `model` — any `ModelInterface` (here, our LJ model)
- `plumed_input` — a list of PLUMED command strings (or a path to a `.dat` file)
- `timestep` — MD timestep in **ps** (must match `integrate()`'s `timestep`)
- `kT` — thermal energy in **eV** (used by well-tempered metadynamics and similar)

The PLUMED commands define collective variables and what to do with them.
Here we just define a distance CV and `PRINT` it to a file — no bias yet.

> **Unit reminder**: PLUMED input files always use PLUMED's native units.
> Distances → nm, energies → kJ/mol, forces → kJ/mol/nm.
"""

# %%
TIMESTEP = 0.001  # ps
TEMPERATURE = 100  # K
kT = TEMPERATURE * 8.617333e-5  # eV (kB * T)

work_dir = Path("01_basics/")
colvar_file = str(work_dir / "COLVAR")

plumed_model = PlumedModel(
    model=lj_model,
    plumed_input=[
        # Define a distance CV between atoms 1 and 2 (PLUMED uses 1-based indexing)
        "d12: DISTANCE ATOMS=1,2",
        # Also compute the angle at atom 2 formed by atoms 1-2-3
        "a123: ANGLE ATOMS=1,2,3",
        # Print both CVs every step
        f"PRINT ARG=d12,a123 STRIDE=1 FILE={colvar_file}",
    ],
    timestep=TIMESTEP,
    kT=kT,
)

print("PlumedModel created.")
print(f"Wrapped model device: {plumed_model.device}")
print(f"Wrapped model dtype:  {plumed_model.dtype}")

# %% [markdown]
"""
## 3. Running a Short Simulation

We run 200 steps of NVT Langevin dynamics with the `PlumedModel`.
PLUMED is initialised lazily on the first `forward()` call, so the first step
may take slightly longer.

The `timestep` passed to `integrate()` **must match** the one passed to `PlumedModel`.
PLUMED uses it to maintain its internal clock for actions like metadynamics.
"""

# %%
N_STEPS = 50

final_state = ts.integrate(
    system=ar_atoms,
    model=plumed_model,
    integrator=ts.Integrator.nvt_langevin,
    n_steps=N_STEPS,
    temperature=TEMPERATURE,
    timestep=TIMESTEP,
)

print(f"Simulation finished. Final step counter: {plumed_model.step}")
print(f"Final positions shape: {final_state.positions.shape}")

# %% [markdown]
"""
## 4. Reading the COLVAR File

PLUMED writes CVs to the `COLVAR` file every `STRIDE` steps. The file has a
commented header line identifying each column.
"""
del plumed_model
# import time
# time.sleep(15) # wait for file to be flushed to disk
# %%
colvar_data = np.loadtxt(colvar_file, comments="#")
print(colvar_data.shape)
steps = colvar_data[:, 0]   # time in ps
d12 = colvar_data[:, 1]     # distance in nm
a123 = colvar_data[:, 2]    # angle in radians

print(f"COLVAR entries: {len(steps)}")
print(f"d12  — mean: {d12.mean():.4f} nm  ({d12.mean()*10:.3f} Å)")
print(f"a123 — mean: {np.degrees(a123).mean():.1f}°")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
ax1.plot(steps, d12 * 10, color="steelblue")     # convert nm → Å for display
ax1.set_ylabel("d₁₂ (Å)")
ax1.axhline(SIGMA * 2**(1/6), color="gray", ls="--", label="LJ minimum")
ax1.legend(fontsize=8)

ax2.plot(steps, np.degrees(a123), color="coral")
ax2.set_ylabel("∠₁₂₃ (°)")
ax2.set_xlabel("Time (ps)")

fig.suptitle("Collective variables — unbiased Ar NVT")
fig.tight_layout()
plt.show()

# %% [markdown]
"""
## 5. File-Based PLUMED Input

For complex setups it is easier to keep the PLUMED commands in a `.dat` file.
Pass its path (as a `str` or `pathlib.Path`) to `plumed_input`:
"""

# %%
dat_file = work_dir / "plumed.dat"
colvar_file2 = str(work_dir / "COLVAR2")

dat_file.write_text(f"""\
# Standard PLUMED input file
d12: DISTANCE ATOMS=1,2
PRINT ARG=d12 STRIDE=1 FILE={colvar_file2}
""")

plumed_model2 = PlumedModel(
    model=lj_model,
    plumed_input=dat_file,          # pass a Path object
    timestep=TIMESTEP,
    kT=kT,
)

_ = ts.integrate(
    system=ar_atoms,
    model=plumed_model2,
    integrator=ts.Integrator.nvt_langevin,
    n_steps=50,
    temperature=TEMPERATURE,
    timestep=TIMESTEP,
)
print("File-based input: OK")

# %% [markdown]
"""
## 6. The Step Counter and Restarts

`PlumedModel.step` tracks how many `forward()` calls have been made. PLUMED uses this
as its internal clock — metadynamics deposits hills at multiples of `PACE`, PRINT
writes at multiples of `STRIDE`, etc.

When **restarting** a simulation from a checkpoint, set `plumed_model.step` to the
last step of the previous run, and pass `restart=True` to the constructor so PLUMED
reads existing output files instead of overwriting them.
"""

# # %%
# print(f"Step counter after first run:  {plumed_model.step}")
# print(f"Step counter after second run: {plumed_model2.step}")

# # Simulate restarting from step 200
# plumed_model.step = 200
# print(f"After manual reset: {plumed_model.step}")

# # %% [markdown]
# """
# ## 7. Inspecting PlumedModel Properties

# `PlumedModel` delegates `device`, `dtype`, `compute_stress`, and `compute_forces`
# to the wrapped model, so it is a drop-in replacement anywhere a `ModelInterface`
# is expected.
# """

# # %%
# print(f"device:          {plumed_model.device}")
# print(f"dtype:           {plumed_model.dtype}")
# print(f"compute_forces:  {plumed_model.compute_forces}")
# print(f"compute_stress:  {plumed_model.compute_stress}")
# print(f"memory_scales_with: {plumed_model.memory_scales_with}")

# # %% [markdown]
# """
# ## Summary

# - `PlumedModel` wraps any `ModelInterface` and adds PLUMED bias transparently.
# - PLUMED input commands use **PLUMED's native units** (nm, kJ/mol, ps).
# - Pass the same `timestep` to both `PlumedModel` and `integrate()`.
# - Use `plumed_model.step` to sync the step counter when restarting.
# - The COLVAR file is a plain text file readable with `numpy.loadtxt`.

# Next: [02_restraints.py](02_restraints.py) — applying harmonic restraints for
# umbrella sampling.
# """
