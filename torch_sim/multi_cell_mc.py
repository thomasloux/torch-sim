"""Multi-Cell Monte Carlo for phase prediction.

This module implements the Multi-Cell Monte Carlo (MC)² algorithm from:
Niu, C., Rao, Y., Windl, W. et al. Multi-cell Monte Carlo method for phase
prediction. npj Comput Mater 5, 120 (2019). https://doi.org/10.1038/s41524-019-0259-z

The algorithm predicts stable phases in chemically complex crystalline systems by
simulating multiple cells representing different phases, coupled through the lever rule
to maintain stoichiometry.

Examples:
    >>> import torch_sim as ts
    >>> from ase.build import bulk
    >>>
    >>> # Create initial cells for different phases
    >>> si_cell = ts.initialize_state(bulk("Si", "diamond", cubic=True), device, dtype)
    >>> sio2_cell = ts.initialize_state(bulk("SiO2", a=4.91, c=5.41), device, dtype)
    >>>
    >>> # Run phase prediction
    >>> final_state = ts.multi_cell_mc_optimize(
    ...     cells=[si_cell, sio2_cell],
    ...     model=model,
    ...     target_composition={14: 1.0, 8: 2.0},  # SiO2 stoichiometry
    ...     n_steps=10000,
    ...     temperature=300.0,
    ... )
    >>>
    >>> # Analyze results
    >>> phases = ts.analyze_phases(final_state)
    >>> print(f"Stable phases: {phases['stable_phases']}")
    >>> print(f"Molar ratios: {phases['molar_ratios']}")
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch


if TYPE_CHECKING:
    from torch_sim.models.interface import ModelInterface
    from torch_sim.state import SimState


@dataclass(kw_only=True)
class MultiCellMCState:
    """State for Multi-Cell Monte Carlo phase prediction.

    Unlike other torch-sim states, this does NOT inherit from SimState
    because it represents multiple independent cells coupled through
    thermodynamic constraints, not a single batched system.

    Attributes:
        cells: List of independent SimState objects representing different phases
        molar_ratios: Virtual percentages for each cell, shape [n_cells], sum to 1.0
        target_composition: Target stoichiometry as {atomic_number: count}
        energies: Per-cell energies, shape [n_cells]
        total_energy: Weighted sum: E_total = Σᵢ xᵢ Eᵢ (scalar)
        n_accepted: Number of accepted moves per cell, shape [n_cells]
        n_proposed: Number of proposed moves per cell, shape [n_cells]
        step_counter: Total number of MC steps taken
        last_flip_cell_idx: Index of cell modified in last move
        last_flip_atom_idx: Index of atom modified in last move
        last_flip_old_Z: Old atomic number before last flip
        last_flip_new_Z: New atomic number after last flip
        enforce_composition_constraint: If True, proposed moves that would make
            the lever-rule target unsatisfiable with non-negative molar ratios
            are rejected before the energy evaluation (hard constraint).
            If False, the solver clamps negative ratios to a small positive
            epsilon and renormalises (soft constraint / original behaviour).
    """

    cells: list[SimState]
    molar_ratios: torch.Tensor
    target_composition: dict[int, float]
    energies: torch.Tensor
    total_energy: torch.Tensor
    n_accepted: torch.Tensor
    n_proposed: torch.Tensor
    step_counter: int
    last_flip_cell_idx: int
    last_flip_atom_idx: int
    last_flip_old_Z: int
    last_flip_new_Z: int
    enforce_composition_constraint: bool = True

    @property
    def n_cells(self) -> int:
        """Number of cells in the simulation."""
        return len(self.cells)

    @property
    def acceptance_rates(self) -> torch.Tensor:
        """Calculate acceptance rate per cell.

        Returns:
            Tensor of shape [n_cells] with acceptance rates in [0, 1]
        """
        return self.n_accepted / torch.clamp(self.n_proposed, min=1.0)


def solve_lever_rule(
    cells: list[SimState],
    target_composition: dict[int, float],
) -> torch.Tensor:
    """Solve for molar ratios that maintain stoichiometry via the lever rule.

    Solves the matrix equation A·X = B where:
    - A[i,j] = count of element i in cell j (composition matrix)
    - X = molar ratios (what we solve for)
    - B = target composition vector

    The lever rule ensures that the weighted average composition across all cells
    matches the target composition.

    Args:
        cells: List of SimState objects representing different phases
        target_composition: Target stoichiometry as {atomic_number: count}

    Returns:
        Molar ratios tensor of shape [n_cells], normalized to sum to 1.0

    Raises:
        ValueError: If the system cannot satisfy the composition constraint

    Notes:
        - Uses torch.linalg.lstsq for robustness (handles over/underdetermined systems)
        - Clamps negative values to small positive epsilon (1e-6)
        - Renormalizes to ensure sum equals 1.0
        - Warns if any molar ratio is very small (< 1e-4), indicating unstable phase

    Examples:
        >>> # Two cells: pure Si and pure SiO2
        >>> si_cell = ts.initialize_state(bulk("Si"), device, dtype)
        >>> sio2_cell = ts.initialize_state(quartz, device, dtype)
        >>> target = {14: 1.0, 8: 2.0}  # Want SiO2 composition
        >>> ratios = solve_lever_rule([si_cell, sio2_cell], target)
        >>> # Should give high ratio for SiO2 cell, low for Si cell
    """
    n_cells = len(cells)
    if n_cells == 0:
        raise ValueError("Must provide at least one cell")

    # Get device and dtype from first cell
    device = cells[0].device
    dtype = cells[0].dtype

    # Extract unique elements across all cells and target
    all_elements = set(target_composition.keys())
    for cell in cells:
        all_elements.update(cell.atomic_numbers.unique().tolist())

    elements = sorted(all_elements)
    n_elements = len(elements)

    # Build composition matrix A: A[i, j] = count of element i in cell j
    A = torch.zeros((n_elements, n_cells), dtype=dtype, device=device)
    for j, cell in enumerate(cells):
        for i, Z in enumerate(elements):
            count = (cell.atomic_numbers == Z).sum().item()
            A[i, j] = float(count)

    # Build target composition vector B
    B = torch.tensor(
        [target_composition.get(Z, 0.0) for Z in elements],
        dtype=dtype,
        device=device,
    )

    # Solve using least squares (handles over/underdetermined cases)
    try:
        solution = torch.linalg.lstsq(A, B, rcond=None)
        molar_ratios = solution.solution
    except torch.linalg.LinAlgError as e:
        raise ValueError(
            f"Failed to solve lever rule: {e}. This may indicate incompatible "
            "cell compositions that cannot satisfy the target composition."
        ) from e

    # Post-process: clamp negatives and renormalize
    molar_ratios = torch.clamp(molar_ratios, min=1e-6)
    molar_ratios = molar_ratios / molar_ratios.sum()

    # Warn about very small molar ratios (potentially unstable phases)
    small_ratio_threshold = 1e-4
    small_ratios = molar_ratios < small_ratio_threshold
    if small_ratios.any():
        small_indices = torch.where(small_ratios)[0].tolist()
        warnings.warn(
            f"Some cells have very low molar ratios (< {small_ratio_threshold}): "
            f"cells {small_indices}. These phases may be unstable.",
            stacklevel=2,
        )

    return molar_ratios


def is_lever_rule_feasible(
    cells: list[SimState],
    target_composition: dict[int, float],
    residual_tol: float = 0.01,
) -> bool:
    """Check whether target_composition is achievable with non-negative molar ratios.

    Uses Non-Negative Least Squares (NNLS) to find the best non-negative
    combination of cell compositions that reproduces the target.  The move
    is considered infeasible when the NNLS residual exceeds
    ``residual_tol * ||B||``, meaning no non-negative weighting of the
    current cells can match the target composition within 1 % (by default).

    This correctly handles cases such as:
    - An element present in the target but absent from all cells after a flip
      (the NNLS residual is large, move is rejected).
    - Cells that are already near the boundary of achievable compositions
      (residual close to zero, move is accepted and the Metropolis check
      then decides the outcome).

    Args:
        cells: Proposed cell configurations after a flip move.
        target_composition: Target stoichiometry as {atomic_number: count}.
        residual_tol: Reject if ``||A·x* - B|| > residual_tol * ||B||``.
            Default 0.01 (1 %).

    Returns:
        True if the NNLS residual is within tolerance (feasible).
    """
    import numpy as np
    from scipy.optimize import nnls

    n_cells = len(cells)

    all_elements = set(target_composition.keys())
    for cell in cells:
        all_elements.update(cell.atomic_numbers.unique().tolist())

    elements = sorted(all_elements)
    n_elements = len(elements)

    A_np = np.zeros((n_elements, n_cells), dtype=np.float64)
    for j, cell in enumerate(cells):
        for i, Z in enumerate(elements):
            count = int((cell.atomic_numbers == Z).sum().item())
            A_np[i, j] = float(count)

    B_np = np.array(
        [target_composition.get(Z, 0.0) for Z in elements], dtype=np.float64
    )

    _, residual = nnls(A_np, B_np)

    b_norm = float(np.linalg.norm(B_np))
    return residual < residual_tol * max(b_norm, 1e-8)


def propose_flip_move(
    state: MultiCellMCState,
    rng: torch.Generator,
) -> tuple[int, int, int]:
    """Propose a flip move: change one atom's species in one cell.

    Selection strategy:
    1. Choose cell weighted by number of atoms (fair sampling)
    2. Choose random atom uniformly in that cell
    3. Choose new element from target_composition with probability
       proportional to target stoichiometry

    Args:
        state: Current MultiCellMCState
        rng: Random number generator for reproducibility

    Returns:
        Tuple of (cell_idx, atom_idx_in_cell, new_atomic_number)

    Notes:
        - Ensures new_Z != current_Z (retries if same element selected)
        - Weighted cell selection ensures fair sampling across different cell sizes
    """
    # Calculate weights based on number of atoms per cell
    n_atoms_per_cell = torch.tensor(
        [cell.n_atoms for cell in state.cells],
        dtype=torch.float32,
        device=state.cells[0].device,
    )

    # Sample cell index weighted by number of atoms
    cell_idx = torch.multinomial(
        n_atoms_per_cell, num_samples=1, replacement=True, generator=rng
    ).item()

    # Sample atom index uniformly within the selected cell
    n_atoms = state.cells[cell_idx].n_atoms
    device = state.cells[0].device
    atom_idx = torch.randint(
        0, n_atoms, (1,), generator=rng, device=device
    ).item()

    # Get current atomic number
    current_Z = state.cells[cell_idx].atomic_numbers[atom_idx].item()

    # Create probability distribution for new element based on target composition
    elements = list(state.target_composition.keys())
    weights = torch.tensor(
        [state.target_composition[Z] for Z in elements],
        dtype=torch.float32,
        device=state.cells[0].device,
    )

    # Sample new element, retry if same as current
    max_retries = 100
    for _ in range(max_retries):
        new_Z_idx = torch.multinomial(
            weights, num_samples=1, replacement=True, generator=rng
        ).item()
        new_Z = elements[new_Z_idx]
        if new_Z != current_Z:
            break
    else:
        # If all retries failed (e.g., single-element system), just pick any other element
        if len(elements) > 1:
            other_elements = [Z for Z in elements if current_Z != Z]
            new_Z = other_elements[0]
        else:
            # Single element system - this shouldn't happen in practice
            new_Z = current_Z

    return cell_idx, atom_idx, new_Z


def evaluate_energy(
    cells: list[SimState],
    model: ModelInterface,
    molar_ratios: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculate per-cell and weighted total energies.

    Uses batched forward pass for GPU efficiency:
    1. Concatenate all cells into single batched state
    2. Single model(batched_state) call
    3. Calculate weighted sum: E_total = Σᵢ xᵢ Eᵢ

    Args:
        cells: List of SimState objects
        model: ModelInterface for energy calculation
        molar_ratios: Molar ratio for each cell, shape [n_cells]

    Returns:
        Tuple of:
        - per_cell_energies: Energy for each cell, shape [n_cells]
        - total_energy: Weighted sum E_total = Σᵢ xᵢ Eᵢ (scalar)

    Examples:
        >>> energies, total = evaluate_energy(cells, model, molar_ratios)
        >>> print(f"Cell energies: {energies}")
        >>> print(f"Weighted total: {total}")
    """
    import torch_sim as ts

    # Concatenate all cells into single batched state for efficient evaluation
    batched_state = ts.concatenate_states(cells)

    # Single batched forward pass
    output = model(batched_state)
    per_cell_energies = output["energy"]  # Shape: [n_cells]

    # Calculate weighted total energy
    total_energy = torch.sum(molar_ratios * per_cell_energies)

    return per_cell_energies, total_energy


def multi_cell_mc_init(
    cells: list[SimState],
    model: ModelInterface,
    *,
    target_composition: dict[int, float],
    enforce_composition_constraint: bool = True,
    **_kwargs: Any,
) -> MultiCellMCState:
    """Initialize multi-cell MC state.

    Steps:
    1. Validate all cells on same device/dtype
    2. Compute initial per-cell energies (batched)
    3. Solve lever rule for initial molar ratios
    4. Compute weighted total energy
    5. Initialize statistics counters

    Args:
        cells: List of SimState objects representing different phases
        model: ModelInterface for energy calculation
        target_composition: Target stoichiometry as {atomic_number: count}
        enforce_composition_constraint: If True (default), proposed moves that
            would make the lever-rule target unsatisfiable with non-negative
            molar ratios are rejected immediately (hard constraint, prevents
            lever-rule degeneracy). If False, the original soft-clamping
            behaviour is used: negative ratios are clamped to a small positive
            epsilon and renormalised, which can lead to one cell absorbing all
            weight when compositions drift to extremes.
        **_kwargs: Additional keyword arguments (ignored, for compatibility)

    Returns:
        Initialized MultiCellMCState

    Raises:
        ValueError: If validation fails (incompatible devices, empty cells, etc.)

    Examples:
        >>> state = multi_cell_mc_init(
        ...     cells=[si_cell, sio2_cell],
        ...     model=model,
        ...     target_composition={14: 1.0, 8: 2.0},
        ...     enforce_composition_constraint=True,
        ... )
    """
    # Validation
    if len(cells) < 2:
        raise ValueError(f"Must provide at least 2 cells, got {len(cells)}")

    if not all(cell.n_atoms > 0 for cell in cells):
        raise ValueError("All cells must have at least one atom")

    # Check device/dtype consistency
    device = cells[0].device
    dtype = cells[0].dtype
    for i, cell in enumerate(cells[1:], start=1):
        if cell.device != device:
            raise ValueError(
                f"All cells must be on same device. "
                f"Cell 0: {device}, Cell {i}: {cell.device}"
            )
        if cell.dtype != dtype:
            raise ValueError(
                f"All cells must have same dtype. Cell 0: {dtype}, Cell {i}: {cell.dtype}"
            )

    # Check that target composition elements exist in at least one cell
    all_elements = set()
    for cell in cells:
        all_elements.update(cell.atomic_numbers.unique().tolist())

    target_elements = set(target_composition.keys())
    if not target_elements.issubset(all_elements):
        missing = target_elements - all_elements
        raise ValueError(
            f"Target composition contains elements not present in any cell: {missing}"
        )

    # Solve lever rule for initial molar ratios
    molar_ratios = solve_lever_rule(cells, target_composition)

    # Evaluate initial energies
    energies, total_energy = evaluate_energy(cells, model, molar_ratios)

    # Initialize statistics
    n_cells = len(cells)
    tensor_kwargs = {"device": device, "dtype": dtype}

    return MultiCellMCState(
        cells=cells,
        molar_ratios=molar_ratios,
        target_composition=target_composition,
        energies=energies,
        total_energy=total_energy,
        n_accepted=torch.zeros(n_cells, **tensor_kwargs),
        n_proposed=torch.zeros(n_cells, **tensor_kwargs),
        step_counter=0,
        last_flip_cell_idx=-1,
        last_flip_atom_idx=-1,
        last_flip_old_Z=-1,
        last_flip_new_Z=-1,
        enforce_composition_constraint=enforce_composition_constraint,
    )


def multi_cell_mc_step(
    state: MultiCellMCState,
    model: ModelInterface,
    *,
    kT: float,
    rng: torch.Generator | None = None,
    **_kwargs: Any,
) -> MultiCellMCState:
    """Perform one multi-cell MC step.

    Steps:
    1. Propose flip move (cell, atom, new_Z)
    2. Clone affected cell and modify atomic_numbers
    3. Feasibility check: reject immediately if lever rule has no non-negative
       solution (composition constraint would be violated)
    4. Solve lever rule for new molar ratios
    5. Evaluate energies for new configuration
    6. Apply Metropolis criterion: P = min(1, exp(-β ΔE_total))
    7. If accepted: keep changes, update statistics
       If rejected: revert to old state, update statistics
    8. Return updated state

    Args:
        state: Current MultiCellMCState
        model: ModelInterface for energy calculation
        kT: Temperature in energy units (e.g., 0.1 * units.energy)
        rng: Random number generator for reproducibility
        **_kwargs: Additional keyword arguments (ignored, for compatibility)

    Returns:
        Updated MultiCellMCState after one MC step

    Examples:
        >>> for _ in range(1000):
        ...     state = multi_cell_mc_step(state, model, kT=0.1)
    """
    from torch_sim.monte_carlo import metropolis_criterion

    # Create RNG if not provided
    if rng is None:
        rng = torch.Generator(device=state.cells[0].device)

    # Propose flip move
    cell_idx, atom_idx, new_Z = propose_flip_move(state, rng)
    old_Z = state.cells[cell_idx].atomic_numbers[atom_idx].item()

    # Clone cells and modify the affected one
    new_cells = [
        cell if i != cell_idx else cell.clone()
        for i, cell in enumerate(state.cells)
    ]
    new_cells[cell_idx].atomic_numbers[atom_idx] = new_Z

    # Feasibility guard (hard constraint): reject immediately if the proposed
    # composition cannot be satisfied with non-negative molar ratios.
    # Only active when state.enforce_composition_constraint is True.
    if state.enforce_composition_constraint and not is_lever_rule_feasible(
        new_cells, state.target_composition
    ):
        new_n_proposed = state.n_proposed.clone()
        new_n_proposed[cell_idx] += 1
        return MultiCellMCState(
            cells=state.cells,
            molar_ratios=state.molar_ratios,
            target_composition=state.target_composition,
            energies=state.energies,
            total_energy=state.total_energy,
            n_accepted=state.n_accepted.clone(),
            n_proposed=new_n_proposed,
            step_counter=state.step_counter + 1,
            last_flip_cell_idx=cell_idx,
            last_flip_atom_idx=atom_idx,
            last_flip_old_Z=old_Z,
            last_flip_new_Z=new_Z,
            enforce_composition_constraint=state.enforce_composition_constraint,
        )

    # Solve lever rule for new configuration
    new_molar_ratios = solve_lever_rule(new_cells, state.target_composition)

    # Evaluate energies for new configuration
    new_energies, new_total_energy = evaluate_energy(new_cells, model, new_molar_ratios)

    # Apply Metropolis criterion
    # Create tensors for metropolis_criterion (expects shape [batch_size])
    energy_old_tensor = state.total_energy.unsqueeze(0)
    energy_new_tensor = new_total_energy.unsqueeze(0)

    accepted = metropolis_criterion(
        len(state.cells)*energy_new_tensor,  len(state.cells)*energy_old_tensor, kT, rng=rng
    )[0].item()

    # Update statistics
    new_n_proposed = state.n_proposed.clone()
    new_n_proposed[cell_idx] += 1

    new_n_accepted = state.n_accepted.clone()
    if accepted:
        new_n_accepted[cell_idx] += 1

    # Return updated state
    if accepted:
        return MultiCellMCState(
            cells=new_cells,
            molar_ratios=new_molar_ratios,
            target_composition=state.target_composition,
            energies=new_energies,
            total_energy=new_total_energy,
            n_accepted=new_n_accepted,
            n_proposed=new_n_proposed,
            step_counter=state.step_counter + 1,
            last_flip_cell_idx=cell_idx,
            last_flip_atom_idx=atom_idx,
            last_flip_old_Z=old_Z,
            last_flip_new_Z=new_Z,
            enforce_composition_constraint=state.enforce_composition_constraint,
        )
    # Rejected: keep old state but update statistics
    return MultiCellMCState(
        cells=state.cells,
        molar_ratios=state.molar_ratios,
        target_composition=state.target_composition,
        energies=state.energies,
        total_energy=state.total_energy,
        n_accepted=new_n_accepted,
        n_proposed=new_n_proposed,
        step_counter=state.step_counter + 1,
        last_flip_cell_idx=cell_idx,
        last_flip_atom_idx=atom_idx,
        last_flip_old_Z=old_Z,
        last_flip_new_Z=new_Z,
        enforce_composition_constraint=state.enforce_composition_constraint,
    )


def check_convergence(
    molar_ratio_history: list[torch.Tensor],
    window: int = 100,
    threshold: float = 1e-3,
) -> bool:
    """Check if molar ratios have stabilized.

    Convergence criterion: maximum standard deviation across window < threshold

    Args:
        molar_ratio_history: List of molar ratio tensors over time
        window: Number of recent steps to analyze
        threshold: Maximum std dev for convergence

    Returns:
        True if converged, False otherwise

    Examples:
        >>> history = [state.molar_ratios for state in states]
        >>> if check_convergence(history, window=100, threshold=1e-3):
        ...     print("Converged!")
    """
    if len(molar_ratio_history) < window:
        return False

    # Stack recent history: shape [window, n_cells]
    recent = torch.stack(molar_ratio_history[-window:])

    # Calculate std dev per cell: shape [n_cells]
    std_per_cell = recent.std(dim=0)

    # Check if max std dev is below threshold
    max_std = std_per_cell.max().item()
    return max_std < threshold


def analyze_phases(state: MultiCellMCState, threshold: float = 0.01) -> dict:
    """Extract phase information from converged state.

    Args:
        state: Converged MultiCellMCState
        threshold: Minimum molar ratio to consider a phase stable (default: 0.01)

    Returns:
        Dictionary containing:
        - "stable_phases": List[int] - cell indices with molar_ratio > threshold
        - "molar_ratios": torch.Tensor - final molar ratios
        - "compositions": List[dict] - per-cell composition as {element: count}
        - "energies": torch.Tensor - per-cell energies
        - "acceptance_rates": torch.Tensor - per-cell acceptance rates
        - "total_energy": float - final weighted total energy

    Examples:
        >>> final_state = multi_cell_mc_optimize(...)
        >>> phases = analyze_phases(final_state)
        >>> print(f"Stable phases: {phases['stable_phases']}")
        >>> print(f"Molar ratios: {phases['molar_ratios']}")
    """
    # Identify stable phases
    stable_phases = [
        i for i, ratio in enumerate(state.molar_ratios)
        if ratio.item() > threshold
    ]

    # Extract compositions for each cell
    compositions = []
    for cell in state.cells:
        composition = {}
        unique_Z = cell.atomic_numbers.unique()
        for Z in unique_Z:
            count = (cell.atomic_numbers == Z).sum().item()
            composition[int(Z.item())] = count
        compositions.append(composition)

    return {
        "stable_phases": stable_phases,
        "molar_ratios": state.molar_ratios,
        "compositions": compositions,
        "energies": state.energies,
        "acceptance_rates": state.acceptance_rates,
        "total_energy": state.total_energy.item(),
    }


def multi_cell_mc_optimize(
    cells: list[SimState],
    model: ModelInterface,
    *,
    target_composition: dict[int, float],
    n_steps: int = 10_000,
    temperature: float = 300.0,
    convergence_threshold: float = 1e-3,
    convergence_window: int = 100,
    enforce_composition_constraint: bool = True,
    trajectory_file: str | None = None,
    pbar: bool = True,
    seed: int | None = None,
) -> MultiCellMCState:
    """Run multi-cell Monte Carlo phase prediction.

    Args:
        cells: List of initial cell structures (SimState objects)
        model: Energy model (ModelInterface)
        target_composition: Target stoichiometry as {atomic_number: count}
        n_steps: Number of MC steps to run (default: 10,000)
        temperature: Temperature in Kelvin (default: 300.0)
        convergence_threshold: Std dev threshold for molar ratio stability (default: 1e-3)
        convergence_window: Window size for convergence check (default: 100)
        enforce_composition_constraint: If True (default), moves that would
            make the lever-rule target unsatisfiable are rejected immediately
            (hard constraint). If False, negative molar ratios are clamped
            and renormalised (soft constraint / original behaviour).
        trajectory_file: Optional HDF5 file to save trajectory (not yet implemented)
        pbar: Show progress bar (default: True)
        seed: Random seed for reproducibility (default: None)

    Returns:
        Final MultiCellMCState with converged phases

    Examples:
        >>> import torch_sim as ts
        >>> from ase.build import bulk
        >>>
        >>> # Create initial cells for different phases
        >>> si_atoms = bulk("Si", "diamond", cubic=True)
        >>> si_cell = ts.initialize_state(si_atoms, device, dtype)
        >>> sio2_cell = ts.initialize_state(quartz_structure, device, dtype)
        >>>
        >>> # Run phase prediction
        >>> final_state = ts.multi_cell_mc_optimize(
        ...     cells=[si_cell, sio2_cell],
        ...     model=model,
        ...     target_composition={14: 1.0, 8: 2.0},  # SiO2 stoichiometry
        ...     n_steps=10000,
        ...     temperature=300.0,
        ... )
        >>>
        >>> # Analyze results
        >>> phases = ts.analyze_phases(final_state)
        >>> print(f"Stable phases: {phases['stable_phases']}")
        >>> print(f"Molar ratios: {phases['molar_ratios']}")
    """
    from torch_sim.units import MetalUnits

    # Setup random number generator
    rng = torch.Generator(device=model.device)
    if seed is not None:
        rng.manual_seed(seed)

    # Convert temperature to kT (energy units)
    kT = temperature * MetalUnits.temperature

    # Initialize state
    state = multi_cell_mc_init(
        cells, model,
        target_composition=target_composition,
        enforce_composition_constraint=enforce_composition_constraint,
    )

    # Track molar ratio history for convergence
    molar_ratio_history: list[torch.Tensor] = [state.molar_ratios.clone()]

    # Main loop with optional progress bar
    if pbar:
        try:
            from tqdm import tqdm
            iterator = tqdm(range(n_steps), desc="Multi-cell MC")
        except ImportError:
            iterator = range(n_steps)
            warnings.warn(
                "tqdm not installed, progress bar disabled. "
                "Install with: pip install tqdm",
                stacklevel=2,
            )
    else:
        iterator = range(n_steps)

    converged = False
    for step in iterator:
        # Perform MC step
        state = multi_cell_mc_step(state, model, kT=kT, rng=rng)

        # Track molar ratio history
        molar_ratio_history.append(state.molar_ratios.clone())

        # Check convergence periodically
        if (step + 1) % 10 == 0:  # Check every 10 steps
            converged = check_convergence(
                molar_ratio_history,
                window=convergence_window,
                threshold=convergence_threshold,
            )
            if converged:
                if pbar and hasattr(iterator, "close"):
                    iterator.close()
                print(f"Converged at step {step + 1}")
                break

    if not converged:
        warnings.warn(
            f"Did not converge after {n_steps} steps. "
            f"Consider increasing n_steps or relaxing convergence_threshold.",
            stacklevel=2,
        )

    # TODO: Implement trajectory file writing if trajectory_file is provided
    if trajectory_file is not None:
        warnings.warn(
            "Trajectory file writing not yet implemented. "
            "Ignoring trajectory_file parameter.",
            stacklevel=2,
        )

    return state
