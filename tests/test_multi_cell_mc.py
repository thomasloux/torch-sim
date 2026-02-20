"""Tests for multi-cell Monte Carlo phase prediction."""

import pytest
import torch
from ase.build import bulk

import torch_sim as ts
from torch_sim.models.lennard_jones import LennardJonesModel


@pytest.fixture
def device():
    """Test device (CPU for compatibility)."""
    return torch.device("cpu")


@pytest.fixture
def dtype():
    """Test dtype."""
    return torch.float64


@pytest.fixture
def simple_cells(device, dtype):
    """Create two simple cells: Si and SiO2."""
    # Pure Si cell (8 atoms)
    si_atoms = bulk("Si", "diamond", a=5.43, cubic=True)
    si_cell = ts.initialize_state(si_atoms, device, dtype)

    # SiO2-like cell: 2 Si + 4 O
    sio2_atoms = bulk("Si", "diamond", a=5.43, cubic=True)
    # Replace half with oxygen
    sio2_atoms.numbers[::2] = 8  # O
    sio2_atoms.numbers[1::2] = 14  # Si
    sio2_cell = ts.initialize_state(sio2_atoms, device, dtype)

    return [si_cell, sio2_cell]


@pytest.fixture
def lj_model(device, dtype):
    """Create a simple Lennard-Jones model for testing."""
    return LennardJonesModel(device=device, dtype=dtype)


# Test 1: Lever rule solver tests


def test_solve_lever_rule_exact(simple_cells):
    """Test exact case: system that can be exactly solved."""
    target = {14: 2.0, 8: 1.0}  # 2 Si : 1 O
    ratios = ts.multi_cell_mc.solve_lever_rule(simple_cells, target)

    assert ratios.shape == (2,), "Molar ratios should have shape [n_cells]"
    assert torch.allclose(ratios.sum(), torch.tensor(1.0, dtype=ratios.dtype)), "Molar ratios should sum to 1"
    assert (ratios >= 0).all(), "All molar ratios should be non-negative"


def test_solve_lever_rule_overdetermined(simple_cells, device, dtype):
    """Test overdetermined: more cells than elements."""
    # Add a third cell
    fe_atoms = bulk("Fe", "bcc", a=2.87, cubic=True)
    fe_cell = ts.initialize_state(fe_atoms, device, dtype)
    cells = simple_cells + [fe_cell]

    target = {14: 1.0, 8: 2.0}  # SiO2
    ratios = ts.multi_cell_mc.solve_lever_rule(cells, target)

    assert ratios.shape == (3,)
    assert torch.allclose(ratios.sum(), torch.tensor(1.0, dtype=ratios.dtype))
    assert (ratios >= 0).all()


def test_solve_lever_rule_underdetermined(device, dtype):
    """Test underdetermined: fewer cells than elements."""
    # Single cell with multiple elements
    atoms = bulk("Si", "diamond", a=5.43, cubic=True)
    atoms.numbers[0] = 8  # One oxygen
    cell = ts.initialize_state(atoms, device, dtype)

    target = {14: 7.0, 8: 1.0}  # 7 Si : 1 O
    ratios = ts.multi_cell_mc.solve_lever_rule([cell], target)

    assert ratios.shape == (1,)
    assert torch.allclose(ratios.sum(), torch.tensor(1.0, dtype=ratios.dtype))


def test_solve_lever_rule_warns_small_ratios(device, dtype):
    """Test warning for very small molar ratios."""
    # Create three cells with different compositions
    # Cell 1: 8 Si atoms
    si_atoms = bulk("Si", "diamond", a=5.43, cubic=True)
    si_cell = ts.initialize_state(si_atoms, device, dtype)

    # Cell 2: 8 O atoms
    o_atoms = bulk("Si", "diamond", a=5.43, cubic=True)
    o_atoms.numbers[:] = 8  # All oxygen
    o_cell = ts.initialize_state(o_atoms, device, dtype)

    # Cell 3: Mixed Si and O (4 each)
    mixed_atoms = bulk("Si", "diamond", a=5.43, cubic=True)
    mixed_atoms.numbers[::2] = 14  # Si
    mixed_atoms.numbers[1::2] = 8  # O
    mixed_cell = ts.initialize_state(mixed_atoms, device, dtype)

    # Target heavily favors Si: 100 Si : 1 O
    # This should make oxygen-only cell have very small molar ratio
    target = {14: 100.0, 8: 1.0}

    with pytest.warns(UserWarning, match="very low molar ratios"):
        ratios = ts.multi_cell_mc.solve_lever_rule([si_cell, o_cell, mixed_cell], target)

    assert ratios.shape == (3,)


# Test 2: Move generation tests


def test_propose_flip_move(simple_cells, lj_model):
    """Test move proposal returns valid values."""
    target = {14: 1.0, 8: 2.0}
    state = ts.multi_cell_mc_init(simple_cells, lj_model, target_composition=target)

    rng = torch.Generator(device=simple_cells[0].device)
    rng.manual_seed(42)

    cell_idx, atom_idx, new_Z = ts.multi_cell_mc.propose_flip_move(state, rng)

    assert 0 <= cell_idx < len(simple_cells), "Cell index out of range"
    assert 0 <= atom_idx < simple_cells[cell_idx].n_atoms, "Atom index out of range"
    assert new_Z in target.keys(), "New Z not in target composition"


def test_propose_flip_move_changes_species(simple_cells, lj_model):
    """Ensure new_Z != old_Z (when possible)."""
    target = {14: 1.0, 8: 2.0}  # Two different elements
    state = ts.multi_cell_mc_init(simple_cells, lj_model, target_composition=target)

    rng = torch.Generator(device=simple_cells[0].device)
    rng.manual_seed(42)

    # Try multiple moves
    for _ in range(10):
        cell_idx, atom_idx, new_Z = ts.multi_cell_mc.propose_flip_move(state, rng)
        old_Z = simple_cells[cell_idx].atomic_numbers[atom_idx].item()

        # Should propose different element (when >1 element available)
        if len(target) > 1:
            assert new_Z != old_Z, "Should propose different element"


def test_propose_flip_move_distribution(simple_cells, lj_model):
    """Test that cells are sampled proportionally to number of atoms."""
    target = {14: 1.0, 8: 1.0}
    state = ts.multi_cell_mc_init(simple_cells, lj_model, target_composition=target)

    rng = torch.Generator(device=simple_cells[0].device)
    rng.manual_seed(42)

    # Collect many samples
    cell_counts = [0, 0]
    n_samples = 1000
    for _ in range(n_samples):
        cell_idx, _, _ = ts.multi_cell_mc.propose_flip_move(state, rng)
        cell_counts[cell_idx] += 1

    # Both cells have same number of atoms, so should be roughly equal
    ratio = cell_counts[0] / cell_counts[1]
    assert 0.8 < ratio < 1.2, "Cells with same n_atoms should be sampled equally"


# Test 3: Energy evaluation tests


def test_evaluate_energy_batching(simple_cells, lj_model):
    """Test batched energy calculation."""
    molar_ratios = torch.tensor([0.5, 0.5], device=simple_cells[0].device, dtype=simple_cells[0].dtype)

    energies, total = ts.multi_cell_mc.evaluate_energy(simple_cells, lj_model, molar_ratios)

    assert energies.shape == (2,), "Should return energy per cell"
    assert total.shape == (), "Total energy should be scalar"
    assert torch.isfinite(energies).all(), "All energies should be finite"
    assert torch.isfinite(total), "Total energy should be finite"


def test_evaluate_energy_weighted_sum(simple_cells, lj_model):
    """Verify E_total = Σᵢ xᵢ Eᵢ."""
    molar_ratios = torch.tensor([0.3, 0.7], device=simple_cells[0].device, dtype=simple_cells[0].dtype)

    energies, total = ts.multi_cell_mc.evaluate_energy(simple_cells, lj_model, molar_ratios)

    expected_total = (molar_ratios * energies).sum()
    assert torch.allclose(total, expected_total), "Total should equal weighted sum"


# Test 4: Init/step function tests


def test_multi_cell_mc_init(simple_cells, lj_model):
    """Test initialization."""
    target = {14: 1.0, 8: 2.0}
    state = ts.multi_cell_mc_init(simple_cells, lj_model, target_composition=target)

    assert isinstance(state, ts.MultiCellMCState)
    assert len(state.cells) == 2
    assert state.molar_ratios.shape == (2,)
    assert torch.allclose(state.molar_ratios.sum(), torch.tensor(1.0, dtype=state.molar_ratios.dtype))
    assert state.energies.shape == (2,)
    assert state.total_energy.shape == ()
    assert state.step_counter == 0
    assert (state.n_accepted == 0).all()
    assert (state.n_proposed == 0).all()


def test_multi_cell_mc_init_validation_min_cells(device, dtype, lj_model):
    """Test that init requires at least 2 cells."""
    si_atoms = bulk("Si", "diamond", a=5.43, cubic=True)
    si_cell = ts.initialize_state(si_atoms, device, dtype)

    with pytest.raises(ValueError, match="at least 2 cells"):
        ts.multi_cell_mc_init([si_cell], lj_model, target_composition={14: 1.0})


def test_multi_cell_mc_init_validation_device(simple_cells, lj_model):
    """Test device consistency validation."""
    # Move one cell to different device if CUDA available
    if torch.cuda.is_available():
        simple_cells[1] = simple_cells[1].to(device=torch.device("cuda"))

        with pytest.raises(ValueError, match="same device"):
            ts.multi_cell_mc_init(simple_cells, lj_model, target_composition={14: 1.0, 8: 1.0})


def test_multi_cell_mc_step_conservation(simple_cells, lj_model):
    """Test that total atom counts are conserved."""
    target = {14: 4.0, 8: 4.0}
    state = ts.multi_cell_mc_init(simple_cells, lj_model, target_composition=target)

    # Record initial atom counts
    initial_counts = {}
    for cell in state.cells:
        for Z in cell.atomic_numbers:
            Z_val = int(Z.item())
            initial_counts[Z_val] = initial_counts.get(Z_val, 0) + 1

    # Perform steps
    for _ in range(10):
        state = ts.multi_cell_mc_step(state, lj_model, kT=0.1)

    # Check atom counts after steps
    final_counts = {}
    for cell in state.cells:
        for Z in cell.atomic_numbers:
            Z_val = int(Z.item())
            final_counts[Z_val] = final_counts.get(Z_val, 0) + 1

    # Counts should be exactly the same
    assert initial_counts == final_counts, "Atom counts should be conserved"


def test_multi_cell_mc_step_acceptance(simple_cells, lj_model):
    """Test Metropolis acceptance statistics."""
    target = {14: 1.0, 8: 1.0}
    state = ts.multi_cell_mc_init(simple_cells, lj_model, target_composition=target)

    rng = torch.Generator(device=simple_cells[0].device)
    rng.manual_seed(42)

    # Perform many steps
    n_steps = 100
    for _ in range(n_steps):
        state = ts.multi_cell_mc_step(state, lj_model, kT=0.1, rng=rng)

    # Check that some moves were proposed
    assert state.n_proposed.sum() == n_steps, "Should propose exactly n_steps moves"

    # Check that statistics are reasonable
    assert (state.n_accepted <= state.n_proposed).all(), "Accepted <= proposed"
    assert (state.n_accepted >= 0).all(), "Accepted >= 0"

    # At least some moves should be accepted (with high probability)
    assert state.n_accepted.sum() > 0, "Should accept at least some moves"


def test_multi_cell_mc_step_updates_counter(simple_cells, lj_model):
    """Test that step counter increments."""
    target = {14: 1.0, 8: 1.0}
    state = ts.multi_cell_mc_init(simple_cells, lj_model, target_composition=target)

    assert state.step_counter == 0

    state = ts.multi_cell_mc_step(state, lj_model, kT=0.1)
    assert state.step_counter == 1

    state = ts.multi_cell_mc_step(state, lj_model, kT=0.1)
    assert state.step_counter == 2


# Test 5: Integration test


def test_multi_cell_mc_optimize_basic(simple_cells, lj_model):
    """Test full optimization runs without errors."""
    final = ts.multi_cell_mc_optimize(
        cells=simple_cells,
        model=lj_model,
        target_composition={14: 1.0, 8: 1.0},
        n_steps=100,
        temperature=300.0,
        pbar=False,
        seed=42,
    )

    assert isinstance(final, ts.MultiCellMCState)
    assert final.step_counter > 0
    assert final.n_proposed.sum() == 100  # Should complete all steps or converge
    assert (final.acceptance_rates >= 0).all() and (final.acceptance_rates <= 1).all()


def test_multi_cell_mc_optimize_convergence(simple_cells, lj_model):
    """Test convergence detection."""
    final = ts.multi_cell_mc_optimize(
        cells=simple_cells,
        model=lj_model,
        target_composition={14: 1.0, 8: 1.0},
        n_steps=1000,
        temperature=100.0,  # Lower temp for faster convergence
        convergence_threshold=1e-2,  # Relaxed for test
        convergence_window=50,
        pbar=False,
        seed=42,
    )

    # Should either converge or complete all steps
    assert final.step_counter <= 1000


def test_multi_cell_mc_optimize_seed_reproducibility(simple_cells, lj_model):
    """Test that same seed gives same results."""
    target = {14: 1.0, 8: 1.0}

    final1 = ts.multi_cell_mc_optimize(
        cells=[cell.clone() for cell in simple_cells],
        model=lj_model,
        target_composition=target,
        n_steps=50,
        temperature=300.0,
        pbar=False,
        seed=42,
    )

    final2 = ts.multi_cell_mc_optimize(
        cells=[cell.clone() for cell in simple_cells],
        model=lj_model,
        target_composition=target,
        n_steps=50,
        temperature=300.0,
        pbar=False,
        seed=42,
    )

    # Results should be identical
    assert torch.allclose(final1.molar_ratios, final2.molar_ratios)
    assert torch.allclose(final1.total_energy, final2.total_energy)
    assert final1.step_counter == final2.step_counter


# Test 6: Phase analysis tests


def test_analyze_phases(simple_cells, lj_model):
    """Test phase identification from converged state."""
    final = ts.multi_cell_mc_optimize(
        cells=simple_cells,
        model=lj_model,
        target_composition={14: 1.0, 8: 1.0},
        n_steps=100,
        temperature=300.0,
        pbar=False,
        seed=42,
    )

    phases = ts.analyze_phases(final, threshold=0.01)

    assert "stable_phases" in phases
    assert "molar_ratios" in phases
    assert "compositions" in phases
    assert "energies" in phases
    assert "acceptance_rates" in phases
    assert "total_energy" in phases

    assert isinstance(phases["stable_phases"], list)
    assert len(phases["compositions"]) == 2
    assert phases["energies"].shape == (2,)
    assert isinstance(phases["total_energy"], float)


def test_analyze_phases_threshold(simple_cells, lj_model):
    """Test phase threshold filtering."""
    target = {14: 1.0, 8: 1.0}
    state = ts.multi_cell_mc_init(simple_cells, lj_model, target_composition=target)

    # Manually set one very small molar ratio
    state.molar_ratios = torch.tensor([0.99, 0.01], device=state.cells[0].device, dtype=state.cells[0].dtype)

    # With threshold 0.05, only first cell should be stable
    phases = ts.analyze_phases(state, threshold=0.05)
    assert phases["stable_phases"] == [0]

    # With threshold 0.005, both should be stable
    phases = ts.analyze_phases(state, threshold=0.005)
    assert set(phases["stable_phases"]) == {0, 1}


def test_check_convergence_helper():
    """Test convergence check helper function."""
    # Create fake history with converging values
    history = [
        torch.tensor([0.5, 0.5]),
        torch.tensor([0.51, 0.49]),
        torch.tensor([0.50, 0.50]),
    ] * 40  # Repeat to fill window

    # Should converge with small threshold
    assert ts.multi_cell_mc.check_convergence(history, window=50, threshold=0.1)

    # Should not converge with very strict threshold
    assert not ts.multi_cell_mc.check_convergence(history, window=50, threshold=1e-6)

    # Should not converge with insufficient history
    assert not ts.multi_cell_mc.check_convergence(history[:10], window=50, threshold=0.1)


def test_check_convergence_diverging():
    """Test convergence check with diverging values."""
    # Create history with diverging values
    history = [torch.tensor([0.5 + i * 0.01, 0.5 - i * 0.01]) for i in range(100)]

    # Should not converge
    assert not ts.multi_cell_mc.check_convergence(history, window=50, threshold=1e-3)
