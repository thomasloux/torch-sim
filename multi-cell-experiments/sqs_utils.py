"""SQS (Special Quasi-random Structure) generation utilities.

Generates starting cells for multi-cell MC experiments using icet's
simulated-annealing SQS optimizer.  Each returned structure closely
matches the Warren-Cowley pair-correlation functions of a perfectly
random alloy, avoiding artificial short-range order in the starting
configurations.

Reference: Zunger et al., Phys. Rev. Lett. 65, 353 (1990).
"""

from __future__ import annotations

from ase import Atoms
from ase.build import bulk
from icet import ClusterSpace
from icet.tools.structure_generation import generate_sqs_from_supercells


# ─────────────────────────────────────────────────────────────────────────────
# FCC  (AuPt)
# ─────────────────────────────────────────────────────────────────────────────

def make_fcc_sqs(
    species: list[str],
    a: float,
    concentrations: dict[str, float],
    *,
    repeat: tuple[int, int, int] = (2, 2, 2),
    cutoffs: list[float] | None = None,
    random_seed: int = 42,
) -> Atoms:
    """Generate an FCC SQS cell.

    Args:
        species: Allowed species on the lattice, e.g. ['Au', 'Pt'].
        a: FCC lattice parameter in Angstrom (use Vegard's law for alloys).
        concentrations: Target concentrations e.g. {'Au': 0.75, 'Pt': 0.25}.
        repeat: Supercell repetitions of the 4-atom conventional FCC cell.
            Default (2,2,2) gives 32 atoms.
        cutoffs: Pair/triplet cutoffs for ClusterSpace.
            Defaults to [5.0, 4.5] Å (covers 2nd and 3rd FCC neighbor shells).
        random_seed: Seed for the SQS annealing.

    Returns:
        ASE Atoms object with the SQS structure.
    """
    if cutoffs is None:
        cutoffs = [5.0, 4.5]

    prim = bulk(species[0], "fcc", a=a)
    cs   = ClusterSpace(prim, cutoffs=cutoffs,
                        chemical_symbols=[species])
    sc   = bulk(species[0], "fcc", a=a, cubic=True).repeat(repeat)
    return generate_sqs_from_supercells(
        cs, [sc], concentrations, random_seed=random_seed
    )


# ─────────────────────────────────────────────────────────────────────────────
# HCP  (Hf-Zr, or Hf/Zr sublattice in HfZrTaNb)
# ─────────────────────────────────────────────────────────────────────────────

def make_hcp_sqs(
    species: list[str],
    a: float,
    c: float,
    concentrations: dict[str, float],
    *,
    repeat: tuple[int, int, int] = (3, 3, 2),
    cutoffs: list[float] | None = None,
    random_seed: int = 42,
) -> Atoms:
    """Generate an HCP SQS cell.

    Args:
        species: Allowed species e.g. ['Hf', 'Zr'].
        a: HCP a lattice parameter in Angstrom.
        c: HCP c lattice parameter in Angstrom.
        concentrations: Target concentrations e.g. {'Hf': 0.75, 'Zr': 0.25}.
        repeat: Supercell repetitions of the 2-atom primitive HCP cell.
            Default (3,3,2) gives 36 atoms.
        cutoffs: Defaults to [5.0, 4.5] Å.
        random_seed: Seed for the SQS annealing.

    Returns:
        ASE Atoms object with the SQS structure.
    """
    if cutoffs is None:
        cutoffs = [5.0, 4.5]

    prim  = bulk(species[0], "hcp", a=a, c=c)
    n_sites = len(prim)                            # 2 for HCP
    cs    = ClusterSpace(prim, cutoffs=cutoffs,
                         chemical_symbols=[species] * n_sites)
    sc    = prim.repeat(repeat)
    return generate_sqs_from_supercells(
        cs, [sc], concentrations, random_seed=random_seed
    )


# ─────────────────────────────────────────────────────────────────────────────
# BCC  (Ta/Nb sublattice in HfZrTaNb)
# ─────────────────────────────────────────────────────────────────────────────

def make_bcc_sqs(
    species: list[str],
    a: float,
    concentrations: dict[str, float],
    *,
    repeat: tuple[int, int, int] = (2, 2, 4),
    cutoffs: list[float] | None = None,
    random_seed: int = 42,
) -> Atoms:
    """Generate a BCC SQS cell.

    Args:
        species: Allowed species e.g. ['Nb', 'Ta'].
        a: BCC lattice parameter in Angstrom.
        concentrations: Target concentrations e.g. {'Nb': 0.75, 'Ta': 0.25}.
        repeat: Supercell repetitions of the 2-atom conventional BCC cell.
            Default (2,2,4) gives 32 atoms.
        cutoffs: Defaults to [4.8, 4.0] Å (covers 1st and 2nd BCC shells).
        random_seed: Seed for the SQS annealing.

    Returns:
        ASE Atoms object with the SQS structure.
    """
    if cutoffs is None:
        cutoffs = [4.8, 4.0]

    prim = bulk(species[0], "bcc", a=a)
    cs   = ClusterSpace(prim, cutoffs=cutoffs,
                        chemical_symbols=[species])
    sc   = bulk(species[0], "bcc", a=a, cubic=True).repeat(repeat)
    return generate_sqs_from_supercells(
        cs, [sc], concentrations, random_seed=random_seed
    )
