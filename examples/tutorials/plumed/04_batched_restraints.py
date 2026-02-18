# %%
# /// script
# dependencies = [
#     "torch_sim_atomistic[plumed, io]"
# ]
# ///


# %% [markdown]
"""
# Batched Umbrella Sampling

In `02_restraints.py` we ran umbrella windows **one at a time** — a fresh
`PlumedModel` and a separate `ts.integrate()` call for each restraint centre.
That approach is simple but leaves GPUs under-utilised: while one window is
being simulated, all other windows sit idle.

`PlumedModel` supports **multi-system batching**: a single model wraps one
PLUMED instance *per system*, so all umbrella windows advance simultaneously
in a single `ts.integrate()` call.

```
┌────────────────────────────────────────────────────────────┐
│  ts.integrate(system=batched_state, model=plumed_model)    │
│                                                            │
│  GPU: one batched model call → N force evaluations        │
│  CPU: N×PLUMED bias calls (sequential, cheap)             │
│                                                            │
│  System 0  RESTRAINT AT=0.30 nm  →  COLVAR_w0             │
│  System 1  RESTRAINT AT=0.35 nm  →  COLVAR_w1             │
│  …                                                         │
│  System 4  RESTRAINT AT=0.50 nm  →  COLVAR_w4             │
└────────────────────────────────────────────────────────────┘
```

This tutorial covers:
1. Building a batched state from N copies of the same system.
2. Creating per-system PLUMED input with different restraint centres.
3. Running all windows in one `ts.integrate()` call.
4. Reading per-window COLVAR files and computing a simple PMF.
"""

# %%
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

Same Ar FCC system as in `02_restraints.py`.
"""

# %%
SIGMA = 3.405  # Å
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

TIMESTEP = 0.001  # ps
TEMPERATURE = 100  # K
kT = TEMPERATURE * 8.617333e-5  # eV

r_min = 2 ** (1 / 6) * SIGMA  # LJ pair minimum ≈ 3.82 Å
print(f"LJ pair minimum: {r_min:.3f} Å = {r_min / 10:.4f} nm")

# %% [markdown]
"""
## 2. Building a Batched State

We concatenate N identical copies of the Ar system into a single `SimState`.
`ts.concatenate_states` handles the bookkeeping: each atom is tagged with a
`system_idx` that maps it to its originating window.
"""

# %%
AT_CENTRES_NM = np.linspace(0.30, 0.50, 5)  # restraint centres in nm
N_WINDOWS = len(AT_CENTRES_NM)

ar_state = ts.io.atoms_to_state(ar_atoms, DEVICE, DTYPE)
batched_state = ts.concatenate_states([ar_state] * N_WINDOWS)

print(f"Windows:          {N_WINDOWS}")
print(f"Atoms per window: {ar_state.n_atoms}")
print(f"Total atoms:      {batched_state.n_atoms}")
print(f"n_systems:        {batched_state.n_systems}")

# %% [markdown]
"""
## 3. Per-System PLUMED Input

Each window needs its own PLUMED input with a different `AT` value.  We pass a
`list[list[str]]` — one sub-list per system — so no automatic file-name
suffixing is needed (we name the files explicitly).

> **Atom indexing**: each PLUMED instance sees only the atoms of its own
> system, so `ATOMS=1,2` always refers to the first two atoms of *that window*,
> regardless of the global atom ordering in the batched state.
"""

# %%
work_dir = Path("04_batched_restraints/")
work_dir.mkdir(parents=True, exist_ok=True)

KAPPA_NM = 10000.0  # kJ/mol/nm²

per_system_input = [
    [
        "d: DISTANCE ATOMS=1,2",
        f"RESTRAINT ARG=d AT={at_nm:.4f} KAPPA={KAPPA_NM:.1f}",
        f"PRINT ARG=d STRIDE=1 FILE={work_dir / f'COLVAR_w{i}'}",
    ]
    for i, at_nm in enumerate(AT_CENTRES_NM)
]

plumed_model = PlumedModel(
    model=lj_model,
    plumed_input=per_system_input,
    timestep=TIMESTEP,
    kT=kT,
)

print("Per-system PLUMED inputs:")
for i, at_nm in enumerate(AT_CENTRES_NM):
    print(f"  Window {i}: AT={at_nm:.3f} nm → COLVAR_w{i}")

# %% [markdown]
"""
## 4. Running All Windows in One Call

A single `ts.integrate()` call advances all windows simultaneously.  Under the
hood:

1. The LJ model evaluates forces for the full batched state in one GPU kernel.
2. For each window *i*, PLUMED applies the harmonic bias and records the CV.

The wall-clock time is roughly the same as running a *single* window — not
N × single-window time.
"""

# %%
N_STEPS = 2000

final_state = ts.integrate(
    system=batched_state,
    model=plumed_model,
    integrator=ts.Integrator.nvt_langevin,
    n_steps=N_STEPS,
    temperature=TEMPERATURE,
    timestep=TIMESTEP,
)

print(f"Simulation done. Steps: {plumed_model.step}")
print(f"Final state n_systems: {final_state.n_systems}")

# Flush PLUMED output files by releasing the PLUMED instances
del plumed_model

# %% [markdown]
"""
## 5. Loading and Plotting the Windows

Each per-window COLVAR file is read independently — the API is identical to
the sequential case.
"""

# %%
colvar_paths = [str(work_dir / f"COLVAR_w{i}") for i in range(N_WINDOWS)]

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# Left: time series
ax = axes[0]
for i, (path, at_nm) in enumerate(zip(colvar_paths, AT_CENTRES_NM)):
    data = np.loadtxt(path, comments="#")
    d12_ang = data[:, 1] * 10  # nm → Å
    ax.plot(data[:, 0], d12_ang, alpha=0.7, label=f"AT={at_nm:.2f} nm")
ax.set_xlabel("Time (ps)")
ax.set_ylabel("d₁₂ (Å)")
ax.set_title("Batched umbrella sampling — time series")
ax.legend(fontsize=7)

# Right: distributions
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
## 6. Simple PMF Estimate

The free-energy analysis is identical to `02_restraints.py`:

$$F(\\xi) = -k_B T \\ln P_\\text{biased}(\\xi) - V_\\text{bias}(\\xi) + C$$

The only difference is that all windows were sampled in a single simulation.
"""

# %%
kT_eV = kT
kappa_ev_ang2 = KAPPA_NM / (96.4853321 * 100)  # kJ/mol/nm² → eV/Å²

fig, ax = plt.subplots(figsize=(7, 4))

for i, (path, at_nm) in enumerate(zip(colvar_paths, AT_CENTRES_NM)):
    at_ang = at_nm * 10

    data = np.loadtxt(path, comments="#")
    d12_ang = data[:, 1] * 10
    counts, edges = np.histogram(d12_ang, bins=20, density=True)
    centres = 0.5 * (edges[:-1] + edges[1:])

    with np.errstate(divide="ignore"):
        f_biased = -kT_eV * np.log(np.where(counts > 0, counts, np.nan))

    v_bias = 0.5 * kappa_ev_ang2 * (centres - at_ang) ** 2
    pmf_raw = f_biased - v_bias
    pmf_raw -= np.nanmin(pmf_raw)

    ax.plot(centres, pmf_raw, "o-", ms=3, label=f"Window {i + 1}")

ax.set_xlabel("d₁₂ (Å)")
ax.set_ylabel("PMF (eV, shifted)")
ax.set_title(
    "Per-window PMF from batched umbrella sampling\n"
    "(full WHAM needed for converged result)"
)
ax.legend(fontsize=8)
fig.tight_layout()
plt.show()

# %% [markdown]
"""
## 7. Shared Input with Automatic File Suffixing

If all windows use the *same* PLUMED script (same CVs, same bias parameters,
only the output file changes), you can pass a single `list[str]` and let
`PlumedModel` append `.{i}` to each `FILE=` argument automatically:

```python
plumed_model = PlumedModel(
    model=lj_model,
    plumed_input=[
        "d: DISTANCE ATOMS=1,2",
        "RESTRAINT ARG=d AT=0.38 KAPPA=10000.0",
        "PRINT ARG=d STRIDE=1 FILE=COLVAR",   # → COLVAR.0, COLVAR.1, …
    ],
    timestep=0.001,
    kT=kT,
)
```

This is convenient for simple analysis (e.g. computing the average CV
distribution across many independent replicas), but for umbrella sampling
where each window has a different `AT` you need per-system input.

## Summary

| Approach | Code | When to use |
|----------|------|-------------|
| Sequential (`02_restraints.py`) | One `PlumedModel` + `integrate()` per window | Quick tests, large per-window steps |
| Batched (this tutorial) | One `PlumedModel` with `list[list[str]]` | Many windows, GPU acceleration |
| Shared input | One `list[str]`, auto-suffix | Same bias for all systems |

The post-processing (WHAM, MBAR) is identical in all cases — just point your
analysis tool at the per-window COLVAR files.
"""
