"""Symmetry utilities for crystal structures using moyopy.

Functions operate on single (unbatched) systems. The ``n_ops`` dimension
refers to the number of symmetry operations of the space group.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch


if TYPE_CHECKING:
    from moyopy import MoyoDataset

    from torch_sim.state import SimState


def _moyo_dataset(
    cell: torch.Tensor,
    frac_pos: torch.Tensor,
    atomic_numbers: torch.Tensor,
    symprec: float = 1e-4,
    angle_tolerance: float | None = None,
) -> MoyoDataset:
    """Get MoyoDataset from cell, fractional positions, and atomic numbers."""
    from moyopy import Cell, MoyoDataset

    moyo_cell = Cell(
        basis=cell.detach().cpu().tolist(),
        positions=frac_pos.detach().cpu().tolist(),
        numbers=atomic_numbers.detach().cpu().int().tolist(),
    )
    return MoyoDataset(moyo_cell, symprec=symprec, angle_tolerance=angle_tolerance)


def _extract_symmetry_ops(
    dataset: MoyoDataset, dtype: torch.dtype, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract rotation and translation tensors from a MoyoDataset.

    Returns:
        (rotations, translations) with shapes (n_ops, 3, 3) and (n_ops, 3).
    """
    rotations = torch.as_tensor(
        dataset.operations.rotations, dtype=dtype, device=device
    ).round()
    translations = torch.as_tensor(
        dataset.operations.translations, dtype=dtype, device=device
    )
    return rotations, translations


def get_symmetry_datasets(
    state: SimState,
    symprec: float = 1e-4,
    angle_tolerance: float | None = None,
) -> list[MoyoDataset]:
    """Get MoyoDataset for each system in a SimState."""
    datasets = []
    for single in state.split():
        cell = single.row_vector_cell[0]
        frac = single.positions @ torch.linalg.inv(cell)
        datasets.append(
            _moyo_dataset(cell, frac, single.atomic_numbers, symprec, angle_tolerance)
        )
    return datasets


# Above this threshold, build_symmetry_map falls back to a per-operation loop
# to avoid allocating an O(n_ops * n_atoms^2) tensor that can OOM on supercells.
_SYMM_MAP_CHUNK_THRESHOLD = 200


def build_symmetry_map(
    rotations: torch.Tensor,
    translations: torch.Tensor,
    frac_pos: torch.Tensor,
) -> torch.Tensor:
    """Build atom mapping for each symmetry operation.

    For each (R, t), maps atom i to atom j where R @ frac_i + t ≈ frac_j (mod 1).

    Returns:
        Symmetry mapping tensor, shape (n_ops, n_atoms).
    """
    n_ops = rotations.shape[0]
    n_atoms = frac_pos.shape[0]

    if n_atoms <= _SYMM_MAP_CHUNK_THRESHOLD:
        # Vectorized: allocates (n_ops, n_atoms, n_atoms, 3) — fast for small systems
        # einsum computes R[o] @ frac[n] for all (o, n) pairs at once
        new_pos = torch.einsum("oij,nj->oni", rotations, frac_pos) + translations[:, None]
        delta = frac_pos[None, None] - new_pos[:, :, None]
        delta -= delta.round()
        return torch.argmin(torch.linalg.norm(delta, dim=-1), dim=-1).long()

    # Per-op loop: allocates only (n_atoms, n_atoms, 3) at a time
    # Equivalent to vectorized path: frac @ R.T == R @ frac per row
    result = torch.empty(n_ops, n_atoms, dtype=torch.long, device=frac_pos.device)
    for op_idx in range(n_ops):
        new_pos_op = frac_pos @ rotations[op_idx].T + translations[op_idx]
        delta = frac_pos[None, :, :] - new_pos_op[:, None, :]
        delta -= delta.round()
        result[op_idx] = torch.argmin(torch.linalg.norm(delta, dim=-1), dim=-1)
    return result


def build_symmetry_map_from_primitive_symmetries(
    dataset: MoyoDataset,
    cell: torch.Tensor,
    frac_pos: torch.Tensor,
) -> torch.Tensor:
    """Build atom mapping using moyopy's prim_permutations to narrow search.

    For each symmetry operation (R, t), finds which atom j each atom i maps to
    via ``R @ frac_pos[i] + t ≈ frac_pos[j] (mod 1)``. Equivalent to
    :func:`build_symmetry_map` but exploits the coset/centering decomposition
    exposed by moyopy to reduce the search space.

    Conceptual model
    ~~~~~~~~~~~~~~~~
    The input cell's symmetry operations decompose as centering translations
    composed with point-group (coset) operations::

        for k in range(n_centering):       # centering translations
            for j in range(n_coset):       # point-group operations
                operation[k * n_coset + j] = (R_j, t_k + t_j)

    moyopy exposes two key mappings:

    - ``prim_permutations`` (n_coset, n_prim): permutation of **primitive**
      atom indices under each coset op j. Tells you which primitive orbit
      atom i lands in after applying R_j.
    - ``mapping_std_prim`` (n_atoms,): maps each input atom to its primitive
      atom index. Multiple input atoms share the same primitive atom (one
      per centering translation), so each primitive orbit contains exactly
      ``n_centering`` input atoms.

    Algorithm (3 steps)
    ~~~~~~~~~~~~~~~~~~~
    1. **Base permutations** (n_coset ops): For each coset op j and atom i,
       ``prim_permutations[j, mapping[i]]`` gives the target primitive orbit.
       Only the ``n_centering`` atoms in that orbit are candidates, reducing
       the nearest-neighbor search from O(n_atoms) to O(n_centering) per atom.
       All coset ops are vectorized into a single
       ``(n_coset, n_atoms, n_centering, 3)`` distance computation.

    2. **Centering permutations** (n_centering ops): Centering ops have
       identity rotation (R = I), so ``new_pos = frac_pos + t_k``. Each atom
       stays in its own primitive orbit, so candidates are again only
       ``n_centering`` atoms. Vectorized as
       ``(n_centering, n_atoms, n_centering, 3)``.

    3. **Composition**: The full permutation for operation ``k * n_coset + j``
       is ``centering_perm_k[base_perm_j[i]]`` — a single ``torch.gather``.

    Complexity
    ~~~~~~~~~~
    - Step 1: O(n_coset × n_atoms × n_centering) instead of
      O(n_coset × n_atoms²)
    - Step 2: O(n_centering × n_atoms × n_centering) = O(n_atoms × n_centering²)
      instead of O(n_centering × n_atoms²)
    - Step 3: O(n_ops × n_atoms) — pure indexing

    Best suited for CPU where the reduced FLOPS outweigh kernel launch overhead.
    On GPU, the brute-force :func:`build_symmetry_map` may be faster due to
    better memory coalescing and massive parallelism on the regular O(n_atoms²)
    distance matrix.

    Args:
        dataset: MoyoDataset with ``prim_permutations``, ``mapping_std_prim``,
            and ``operations`` attributes.
        cell: Lattice vectors, shape (3, 3), row-vector convention.
        frac_pos: Fractional coordinates, shape (n_atoms, 3).

    Returns:
        Symmetry mapping tensor, shape (n_ops, n_atoms). Entry ``[o, i]``
        is the index of the atom that atom ``i`` maps to under operation ``o``.
    """
    dtype, device = frac_pos.dtype, frac_pos.device
    ops = dataset.operations
    n_ops = ops.num_operations

    prim_perms = torch.as_tensor(
        dataset.prim_permutations, dtype=torch.long, device=device
    )
    mapping = torch.as_tensor(dataset.mapping_std_prim, dtype=torch.long, device=device)
    rotations = torch.as_tensor(ops.rotations, dtype=dtype, device=device).round()
    translations = torch.as_tensor(ops.translations, dtype=dtype, device=device)

    n_atoms = frac_pos.shape[0]
    n_coset = prim_perms.shape[0]
    n_prim = prim_perms.shape[1]
    n_centering = n_ops // n_coset

    # Candidate index matrix: (n_prim, n_centering) — each orbit has n_centering atoms
    sorted_idx = torch.argsort(mapping)
    cand_matrix = sorted_idx.reshape(n_prim, n_centering)

    # lattice matrix for Cartesian distance (column-vector convention)
    lattice = cell.T

    # Step 1: Base permutations for the n_coset coset representatives.
    # All ops at once: transform positions, gather candidates, find nearest.
    # Peak allocation: (n_coset, n_atoms, n_centering, 3)
    coset_R = rotations[:n_coset]  # (n_coset, 3, 3)
    coset_t = translations[:n_coset]  # (n_coset, 3)
    new_pos = (
        torch.einsum("oij,nj->oni", coset_R, frac_pos) + coset_t[:, None]
    )  # (n_coset, n_atoms, 3)
    target_prim = prim_perms[:, mapping]  # (n_coset, n_atoms)
    candidates = cand_matrix[target_prim]  # (n_coset, n_atoms, n_centering)
    diff = new_pos[:, :, None, :] - frac_pos[candidates]
    diff -= diff.round()
    cart = diff @ lattice.T  # (n_coset, n_atoms, n_centering, 3)
    best = torch.argmin(torch.linalg.norm(cart, dim=-1), dim=-1)  # (n_coset, n_atoms)
    oj = torch.arange(n_coset, device=device)[:, None].expand_as(best)
    ai = torch.arange(n_atoms, device=device)[None, :].expand_as(best)
    base_perms = candidates[oj, ai, best]  # (n_coset, n_atoms)

    # Step 2: Centering permutations. Centering ops have identity rotation,
    # so new_pos = frac_pos + t_k. Each atom stays in its own orbit.
    # Peak allocation: (n_centering, n_atoms, n_centering, 3)
    centering_t = translations[::n_coset]  # (n_centering, 3)
    new_pos_c = frac_pos[None] + centering_t[:, None]  # (n_centering, n_atoms, 3)
    same_orbit = cand_matrix[mapping]  # (n_atoms, n_centering)
    diff_c = new_pos_c[:, :, None, :] - frac_pos[same_orbit][None]
    diff_c -= diff_c.round()
    cart_c = diff_c @ lattice.T  # (n_centering, n_atoms, n_centering, 3)
    best_c = torch.argmin(
        torch.linalg.norm(cart_c, dim=-1), dim=-1
    )  # (n_centering, n_atoms)
    centering_perms = same_orbit[
        torch.arange(n_atoms, device=device).unsqueeze(0).expand_as(best_c),
        best_c,
    ]  # (n_centering, n_atoms)

    # Step 3: Compose — perm[k*n_coset + j] = centering_perm_k[base_perm_j]
    # gather centering_perms at indices given by base_perms
    full_perms = torch.gather(
        centering_perms.unsqueeze(1).expand(-1, n_coset, -1),  # (K, J, N)
        2,
        base_perms.unsqueeze(0).expand(n_centering, -1, -1),  # (K, J, N)
    ).reshape(n_ops, n_atoms)

    return full_perms


def prep_symmetry(
    cell: torch.Tensor,
    positions: torch.Tensor,
    atomic_numbers: torch.Tensor,
    symprec: float = 1e-4,
    angle_tolerance: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get symmetry rotations and atom mappings for a structure.

    Returns:
        (rotations, symm_map) with shapes (n_ops, 3, 3) and (n_ops, n_atoms).
    """
    frac_pos = positions @ torch.linalg.inv(cell)
    dataset = _moyo_dataset(cell, frac_pos, atomic_numbers, symprec, angle_tolerance)
    rotations, translations = _extract_symmetry_ops(dataset, cell.dtype, cell.device)
    return rotations, build_symmetry_map(rotations, translations, frac_pos)


def _refine_symmetry_impl(
    cell: torch.Tensor,
    positions: torch.Tensor,
    atomic_numbers: torch.Tensor,
    symprec: float = 0.01,
    angle_tolerance: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Core refinement returning all intermediate data for reuse.

    Returns:
        (refined_cell, refined_positions, rotations, translations)
    """
    dtype, device = cell.dtype, cell.device
    frac_pos = positions @ torch.linalg.inv(cell)
    dataset = _moyo_dataset(cell, frac_pos, atomic_numbers, symprec, angle_tolerance)
    rotations, translations = _extract_symmetry_ops(dataset, dtype, device)
    n_ops, n_atoms = rotations.shape[0], positions.shape[0]

    # Symmetrize cell metric: g_sym = avg(R^T @ g @ R), then polar decomposition
    metric = cell @ cell.T
    metric_sym = torch.einsum("nji,jk,nkl->il", rotations, metric, rotations) / n_ops

    def _mat_sqrt(mat: torch.Tensor) -> torch.Tensor:
        evals, evecs = torch.linalg.eigh(mat)
        return evecs @ torch.diag(evals.clamp(min=0).sqrt()) @ evecs.T

    new_cell = _mat_sqrt(metric_sym) @ torch.linalg.solve(_mat_sqrt(metric), cell)

    # Symmetrize positions via displacement averaging over symmetry orbits
    new_frac = positions @ torch.linalg.inv(new_cell)
    symm_map = build_symmetry_map(rotations, translations, new_frac)

    transformed = torch.einsum("oij,nj->oni", rotations, new_frac) + translations[:, None]
    disp = transformed - new_frac[symm_map]
    disp -= disp.round()  # wrap into [-0.5, 0.5]

    target = symm_map.reshape(-1).unsqueeze(-1).expand(-1, 3)
    accum = torch.zeros(n_atoms, 3, dtype=dtype, device=device)
    accum.scatter_add_(0, target, disp.reshape(-1, 3))

    new_positions = (new_frac + accum / n_ops) @ new_cell
    return new_cell, new_positions, rotations, translations


def refine_symmetry(
    cell: torch.Tensor,
    positions: torch.Tensor,
    atomic_numbers: torch.Tensor,
    symprec: float = 0.01,
    angle_tolerance: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetrize cell and positions according to the detected space group.

    Uses polar decomposition for the cell metric tensor and scatter-add
    averaging over symmetry orbits for atomic positions.

    Returns:
        (symmetrized_cell, symmetrized_positions) as row vectors.
    """
    new_cell, new_positions, _rotations, _translations = _refine_symmetry_impl(
        cell, positions, atomic_numbers, symprec, angle_tolerance
    )
    return new_cell, new_positions


def refine_and_prep_symmetry(
    cell: torch.Tensor,
    positions: torch.Tensor,
    atomic_numbers: torch.Tensor,
    symprec: float = 0.01,
    angle_tolerance: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Refine symmetry and get ops/mappings in a single moyopy call.

    Combines ``refine_symmetry`` and ``prep_symmetry`` to avoid redundant
    symmetry detection. Used by ``FixSymmetry.from_state``.

    Returns:
        (refined_cell, refined_positions, rotations, symm_map)
    """
    new_cell, new_positions, rotations, translations = _refine_symmetry_impl(
        cell, positions, atomic_numbers, symprec, angle_tolerance
    )
    # Build symm_map on the final refined fractional coordinates
    refined_frac = new_positions @ torch.linalg.inv(new_cell)
    symm_map = build_symmetry_map(rotations, translations, refined_frac)
    return new_cell, new_positions, rotations, symm_map


def symmetrize_rank1(
    lattice: torch.Tensor,
    vectors: torch.Tensor,
    rotations: torch.Tensor,
    symm_map: torch.Tensor,
) -> torch.Tensor:
    """Symmetrize a rank-1 per-atom tensor (forces, velocities, displacements).

    Works in fractional coordinates internally. Returns symmetrized Cartesian tensor.
    """
    n_ops, n_atoms = rotations.shape[0], vectors.shape[0]
    scaled = vectors @ torch.linalg.inv(lattice)
    # Rotate each vector by each symmetry op: scaled @ R^T
    rotated = torch.einsum("ij,nkj->nik", scaled, rotations).reshape(-1, 3)
    # Scatter-add to target atoms and average
    target = symm_map.reshape(-1).unsqueeze(-1).expand(-1, 3)
    accum = torch.zeros(n_atoms, 3, dtype=vectors.dtype, device=vectors.device)
    accum.scatter_add_(0, target, rotated)
    return (accum / n_ops) @ lattice


def symmetrize_rank2(
    lattice: torch.Tensor,
    tensor: torch.Tensor,
    rotations: torch.Tensor,
) -> torch.Tensor:
    """Symmetrize a rank-2 tensor (stress, strain) over all symmetry operations."""
    n_ops = rotations.shape[0]
    inv_lat = torch.linalg.inv(lattice)
    scaled = lattice @ tensor @ lattice.T
    sym_scaled = torch.einsum("nji,jk,nkl->il", rotations, scaled, rotations) / n_ops
    return inv_lat @ sym_scaled @ inv_lat.T
