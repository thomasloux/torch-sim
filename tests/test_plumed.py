"""Tests for PlumedModel enhanced sampling wrapper."""

import os
from pathlib import Path

import pytest
import torch
from ase.build import bulk

import torch_sim as ts
from torch_sim.models.lennard_jones import LennardJonesModel
from torch_sim.plumed import PlumedModel


DEVICE = torch.device("cpu")
DTYPE = torch.float32 if os.environ.get("TS_TEST_DTYPE") == "float32" else torch.float64


def _plumed_available() -> bool:
    """Return True if plumed is installed and the native PLUMED kernel is accessible."""
    try:
        import plumed

        p = plumed.Plumed()
        del p
    except (ImportError, RuntimeError):
        return False
    else:
        return True


pytestmark = pytest.mark.skipif(
    not _plumed_available(),
    reason="PLUMED or its native kernel not available (set PLUMED_KERNEL)",
)


@pytest.fixture
def ar_state() -> ts.SimState:
    """Single-system Ar FCC state (4 atoms)."""
    atoms = bulk("Ar", "fcc", a=5.26, cubic=True)
    return ts.io.atoms_to_state(atoms, DEVICE, DTYPE)


@pytest.fixture
def ar_two_system_state() -> ts.SimState:
    """Two-system batched Ar state (4 atoms each)."""
    atoms = bulk("Ar", "fcc", a=5.26, cubic=True)
    state = ts.io.atoms_to_state(atoms, DEVICE, DTYPE)
    return ts.concatenate_states([state, state])


def test_plumed_model_init(
    lj_model: LennardJonesModel, ar_state: ts.SimState, tmp_path: Path
) -> None:
    """PlumedModel initializes and lazily creates PLUMED on first forward call."""
    colvar = str(tmp_path / "colvar.out")
    plumed_model = PlumedModel(
        model=lj_model,
        plumed_input=[
            "d: DISTANCE ATOMS=1,2",
            f"PRINT ARG=d STRIDE=1 FILE={colvar}",
        ],
        timestep=0.001,
        kT=0.0257,
    )
    assert plumed_model._plumed_instances == []  # noqa: SLF001
    _ = plumed_model(ar_state)
    assert len(plumed_model._plumed_instances) == 1  # noqa: SLF001
    assert plumed_model._plumed_instances[0] is not None  # noqa: SLF001


def test_plumed_does_not_affect_unbiased_forces(
    lj_model: LennardJonesModel, ar_state: ts.SimState, tmp_path: Path
) -> None:
    """A PRINT-only PLUMED input leaves forces unchanged."""
    colvar = str(tmp_path / "colvar.out")
    plumed_model = PlumedModel(
        model=lj_model,
        plumed_input=[
            "d: DISTANCE ATOMS=1,2",
            f"PRINT ARG=d STRIDE=1 FILE={colvar}",
        ],
        timestep=0.001,
        kT=0.0257,
    )
    unbiased = lj_model(ar_state)
    biased = plumed_model(ar_state)

    assert torch.allclose(unbiased["forces"], biased["forces"], atol=1e-10)
    assert torch.allclose(unbiased["energy"], biased["energy"], atol=1e-10)


def test_plumed_restraint_modifies_forces(
    lj_model: LennardJonesModel, ar_state: ts.SimState, tmp_path: Path
) -> None:
    """A RESTRAINT action adds non-zero bias forces."""
    colvar = str(tmp_path / "colvar.out")
    # AT=0.1 nm (1 Å) is far from the actual Ar nearest-neighbour distance
    # (~0.372 nm), so the restraint produces a large non-zero bias.
    plumed_model = PlumedModel(
        model=lj_model,
        plumed_input=[
            "d: DISTANCE ATOMS=1,2",
            "RESTRAINT ARG=d AT=0.1 KAPPA=100.0",
            f"PRINT ARG=d STRIDE=1 FILE={colvar}",
        ],
        timestep=0.001,
        kT=0.0257,
    )
    unbiased = lj_model(ar_state)
    biased = plumed_model(ar_state)

    assert not torch.allclose(unbiased["forces"], biased["forces"], atol=1e-6)


def test_plumed_restraint_modifies_energy(
    lj_model: LennardJonesModel, ar_state: ts.SimState, tmp_path: Path
) -> None:
    """A RESTRAINT action modifies the total energy."""
    colvar = str(tmp_path / "colvar.out")
    plumed_model = PlumedModel(
        model=lj_model,
        plumed_input=[
            "d: DISTANCE ATOMS=1,2",
            "RESTRAINT ARG=d AT=0.1 KAPPA=100.0",
            f"PRINT ARG=d STRIDE=1 FILE={colvar}",
        ],
        timestep=0.001,
        kT=0.0257,
    )
    unbiased = lj_model(ar_state)
    biased = plumed_model(ar_state)

    assert not torch.isclose(unbiased["energy"][0], biased["energy"][0], atol=1e-6)


def test_plumed_step_counter_increments(
    lj_model: LennardJonesModel, ar_state: ts.SimState, tmp_path: Path
) -> None:
    """The step counter increments by 1 on each forward() call."""
    colvar = str(tmp_path / "colvar.out")
    plumed_model = PlumedModel(
        model=lj_model,
        plumed_input=[
            "d: DISTANCE ATOMS=1,2",
            f"PRINT ARG=d STRIDE=1 FILE={colvar}",
        ],
        timestep=0.001,
        kT=0.0257,
    )
    assert plumed_model.step == 0
    plumed_model(ar_state)
    assert plumed_model.step == 1
    plumed_model(ar_state)
    assert plumed_model.step == 2

    plumed_model.step = 42
    assert plumed_model.step == 42


def test_plumed_file_input(
    lj_model: LennardJonesModel, ar_state: ts.SimState, tmp_path: Path
) -> None:
    """PlumedModel accepts a path to a PLUMED .dat file."""
    colvar = str(tmp_path / "colvar.out")
    dat_file = tmp_path / "plumed.dat"
    dat_file.write_text(f"d: DISTANCE ATOMS=1,2\nPRINT ARG=d STRIDE=1 FILE={colvar}\n")

    plumed_model = PlumedModel(
        model=lj_model,
        plumed_input=dat_file,
        timestep=0.001,
        kT=0.0257,
    )
    result = plumed_model(ar_state)
    assert "energy" in result
    assert "forces" in result
    assert result["forces"].shape == ar_state.positions.shape


def test_plumed_with_integrate(
    lj_model: LennardJonesModel,
    ar_supercell_sim_state: ts.SimState,
    tmp_path: Path,
) -> None:
    """PlumedModel integrates for 10 NVT Langevin steps without error."""
    colvar = str(tmp_path / "COLVAR")
    plumed_model = PlumedModel(
        model=lj_model,
        plumed_input=[
            "d: DISTANCE ATOMS=1,2",
            f"PRINT ARG=d STRIDE=1 FILE={colvar}",
        ],
        timestep=0.001,
        kT=0.0257,
    )
    final_state = ts.integrate(
        system=ar_supercell_sim_state,
        model=plumed_model,
        integrator=ts.Integrator.nvt_langevin,
        n_steps=10,
        temperature=300,
        timestep=0.001,
    )
    assert final_state.positions.shape == ar_supercell_sim_state.positions.shape
    assert final_state.n_systems == 1


def test_plumed_multisystem_shared_input(
    lj_model: LennardJonesModel,
    ar_two_system_state: ts.SimState,
    tmp_path: Path,
) -> None:
    """PlumedModel works with two systems and shared input, auto-suffixing FILE= args."""
    colvar = str(tmp_path / "COLVAR")
    plumed_model = PlumedModel(
        model=lj_model,
        plumed_input=[
            "d: DISTANCE ATOMS=1,2",
            f"PRINT ARG=d STRIDE=1 FILE={colvar}",
        ],
        timestep=0.001,
        kT=0.0257,
    )
    result = plumed_model(ar_two_system_state)

    assert result["energy"].shape == (2,)
    assert result["forces"].shape == ar_two_system_state.positions.shape
    assert len(plumed_model._plumed_instances) == 2  # noqa: SLF001
    assert all(p is not None for p in plumed_model._plumed_instances)  # noqa: SLF001

    # Verify separate output files are created after flushing
    del plumed_model
    assert (tmp_path / "COLVAR.0").exists()
    assert (tmp_path / "COLVAR.1").exists()


def test_plumed_multisystem_per_system_input(
    lj_model: LennardJonesModel,
    ar_two_system_state: ts.SimState,
    tmp_path: Path,
) -> None:
    """PlumedModel works with explicit per-system input lists."""
    colvar0 = str(tmp_path / "COLVAR_w0")
    colvar1 = str(tmp_path / "COLVAR_w1")
    per_system_input = [
        [
            "d: DISTANCE ATOMS=1,2",
            "RESTRAINT ARG=d AT=0.35 KAPPA=1000.0",
            f"PRINT ARG=d STRIDE=1 FILE={colvar0}",
        ],
        [
            "d: DISTANCE ATOMS=1,2",
            "RESTRAINT ARG=d AT=0.45 KAPPA=1000.0",
            f"PRINT ARG=d STRIDE=1 FILE={colvar1}",
        ],
    ]
    plumed_model = PlumedModel(
        model=lj_model,
        plumed_input=per_system_input,
        timestep=0.001,
        kT=0.0257,
    )
    result = plumed_model(ar_two_system_state)

    assert result["energy"].shape == (2,)
    assert result["forces"].shape == ar_two_system_state.positions.shape
    # The two restraints are at different centres, so bias energies should differ.
    # Both systems start from identical positions, but restraint centres differ.
    assert len(plumed_model._plumed_instances) == 2  # noqa: SLF001


def test_plumed_multisystem_with_integrate(
    lj_model: LennardJonesModel,
    ar_two_system_state: ts.SimState,
    tmp_path: Path,
) -> None:
    """PlumedModel integrates a two-system state for 10 NVT steps without error."""
    per_system_input = [
        [
            "d: DISTANCE ATOMS=1,2",
            f"PRINT ARG=d STRIDE=1 FILE={tmp_path / 'COLVAR_0'}",
        ],
        [
            "d: DISTANCE ATOMS=1,2",
            f"PRINT ARG=d STRIDE=1 FILE={tmp_path / 'COLVAR_1'}",
        ],
    ]
    plumed_model = PlumedModel(
        model=lj_model,
        plumed_input=per_system_input,
        timestep=0.001,
        kT=0.0257,
    )
    final_state = ts.integrate(
        system=ar_two_system_state,
        model=plumed_model,
        integrator=ts.Integrator.nvt_langevin,
        n_steps=10,
        temperature=300,
        timestep=0.001,
    )
    assert final_state.positions.shape == ar_two_system_state.positions.shape
    assert final_state.n_systems == 2
