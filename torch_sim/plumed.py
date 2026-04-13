"""PLUMED enhanced sampling integration for TorchSim.

This module provides ``PlumedModel``, a model wrapper that integrates PLUMED
enhanced sampling (metadynamics, umbrella sampling, collective variables, etc.)
with any TorchSim ``ModelInterface``.

Example::

    import torch_sim as ts

    plumed_model = ts.PlumedModel(
        model=model,
        plumed_input=[
            "d: DISTANCE ATOMS=1,2",
            "METAD ARG=d SIGMA=0.1 HEIGHT=0.05 PACE=100 FILE=HILLS",
            "PRINT ARG=d STRIDE=10 FILE=COLVAR",
        ],
        timestep=0.001,
        kT=0.0257,
    )

    final_state = ts.integrate(
        system=atoms,
        model=plumed_model,
        integrator=ts.Integrator.nvt_langevin,
        n_steps=10_000,
        temperature=300,
        timestep=0.001,
    )

Notes:
    PLUMED input is always interpreted in PLUMED's internal units (nm, kJ/mol, ps).
    TorchSim positions, forces, and energies are automatically converted via the
    unit factors set by ``setMDLengthUnits``, ``setMDEnergyUnits``, etc.

    Multi-system batching is supported: one PLUMED instance is created per system.
    When using shared input (a single ``list[str]`` or ``.dat`` file path), output
    file names (``FILE=`` arguments) are automatically suffixed with ``.{i}`` to
    keep each system's output separate.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from torch_sim.models.interface import ModelInterface
from torch_sim.state import SimState  # noqa: TC001


if TYPE_CHECKING:
    from torch_sim.plumed_builder import BiasAction
    from torch_sim.typing import MemoryScaling


# Unit conversion factors for PLUMED's setMDXxxUnits commands.
# setMDXxxUnits(x) means "1 MD engine unit = x PLUMED internal units".
# PLUMED internal units: nm, kJ/mol, ps, amu.
# TorchSim native units: Å, eV, ps, amu.
_PLUMED_LENGTH_FACTOR: float = 0.1  # 1 Å = 0.1 nm
_PLUMED_ENERGY_FACTOR: float = 96.4853321  # 1 eV = 96.4853321 kJ/mol
_PLUMED_TIME_FACTOR: float = 1.0  # 1 ps = 1 ps
_PLUMED_MASS_FACTOR: float = 1.0  # 1 amu = 1 amu


def _is_action_list(inp: object) -> bool:
    """Return True if *inp* looks like a flat list of BiasAction objects."""
    return (
        isinstance(inp, list)
        and bool(inp)
        and hasattr(inp[0], "to_plumed_lines")
        and hasattr(inp[0], "collect_cvs")
    )


def _is_per_system_action_list(inp: object) -> bool:
    """Return True if *inp* looks like a per-system list of BiasAction lists."""
    return (
        isinstance(inp, list)
        and bool(inp)
        and isinstance(inp[0], list)
        and bool(inp[0])
        and hasattr(inp[0][0], "to_plumed_lines")
        and hasattr(inp[0][0], "collect_cvs")
    )


def _normalize_plumed_input(
    plumed_input: list[str]
    | list[list[str]]
    | str
    | Path
    | list[BiasAction]
    | list[list[BiasAction]],
) -> list[str] | list[list[str]] | str | Path:
    """Normalize a plumed_input that may contain BiasAction lists to strings.

    If *plumed_input* is a flat list of :class:`~torch_sim.plumed_builder.BiasAction`
    objects, it is converted to a ``list[str]`` via
    :func:`~torch_sim.plumed_builder.build_plumed_input`.  A per-system list
    of action lists is converted to a ``list[list[str]]``.  All other input
    types are returned unchanged.

    Args:
        plumed_input: PLUMED input in any accepted form.

    Returns:
        list[str] | list[list[str]] | str | Path: Normalised input with no
        :class:`~torch_sim.plumed_builder.BiasAction` objects remaining.
    """
    if _is_action_list(plumed_input):
        from torch_sim.plumed_builder import build_plumed_input

        return build_plumed_input(plumed_input)  # type: ignore[arg-type]
    if _is_per_system_action_list(plumed_input):
        from torch_sim.plumed_builder import build_plumed_input

        return [
            build_plumed_input(actions)  # type: ignore[invalid-argument-type]
            for actions in plumed_input  # type: ignore[not-iterable]
        ]
    return plumed_input  # type: ignore[return-value]


def _load_plumed_lines(plumed_input: list[str] | str | Path) -> list[str]:
    """Load shared PLUMED input into a list of command strings.

    Args:
        plumed_input: Either a list of PLUMED command strings or a path to a
            ``.dat`` file.

    Returns:
        list[str]: Non-empty, non-comment lines from the input.
    """
    if isinstance(plumed_input, (str, Path)):
        return [
            line.strip()
            for line in Path(plumed_input).read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    return list(plumed_input)


def _suffix_file_args(lines: list[str], suffix: str) -> list[str]:
    """Append *suffix* to every ``FILE=<path>`` token in *lines*.

    This is used when running multi-system simulations with shared PLUMED input
    to give each system its own output files.

    Args:
        lines: PLUMED command strings.
        suffix: String appended to every file path, e.g. ``".0"`` or ``".1"``.

    Returns:
        list[str]: Modified lines with suffixed file paths.
    """
    return [re.sub(r"(FILE=)(\S+)", rf"\g<1>\g<2>{suffix}", line) for line in lines]


class PlumedModel(ModelInterface):
    """Wraps a ``ModelInterface`` with PLUMED enhanced sampling bias.

    Calls the underlying model for unbiased energies and forces, then queries
    PLUMED for bias forces and bias energy. The total energy and forces returned
    include both the unbiased model contributions and the PLUMED bias.

    PLUMED is initialized lazily on the first ``forward()`` call. For
    multi-system states, one PLUMED instance is created per system. The
    underlying model is called once per step on the full batched state
    (efficient GPU utilization), then each system's PLUMED instance is
    queried sequentially for its bias.

    When using shared input (``list[str]`` or a ``.dat`` file path) with more
    than one system, output file names (``FILE=`` arguments) are automatically
    suffixed with ``.{i}`` to keep each system's output separate (e.g.
    ``COLVAR`` becomes ``COLVAR.0``, ``COLVAR.1``, …).

    Attributes:
        model: Underlying force/energy model.
        plumed_input: PLUMED commands as a list of strings, a list of
            per-system string lists, or a path to a .dat file.
        timestep: MD timestep in ps.
        kT: Thermal energy in eV.
        log: Path to PLUMED log file; empty string means stdout.
        restart: Whether to restart PLUMED from a previous run.

    Examples:
        Single-system metadynamics::

            plumed_model = PlumedModel(
                model=lj_model,
                plumed_input=[
                    "d: DISTANCE ATOMS=1,2",
                    "METAD ARG=d SIGMA=0.1 HEIGHT=0.05 PACE=100 FILE=HILLS",
                    "PRINT ARG=d STRIDE=10 FILE=COLVAR",
                ],
                timestep=0.001,
                kT=0.0257,
            )

        Multi-system umbrella sampling with per-system input::

            at_values = [0.30, 0.35, 0.40, 0.45, 0.50]  # nm
            per_system_input = [
                [
                    "d: DISTANCE ATOMS=1,2",
                    f"RESTRAINT ARG=d AT={at:.4f} KAPPA=10000.0",
                    f"PRINT ARG=d STRIDE=1 FILE=COLVAR_w{i}",
                ]
                for i, at in enumerate(at_values)
            ]
            plumed_model = PlumedModel(
                model=lj_model,
                plumed_input=per_system_input,
                timestep=0.001,
                kT=0.0086,
            )
    """

    def __init__(
        self,
        model: ModelInterface,
        plumed_input: list[str]
        | list[list[str]]
        | str
        | Path
        | list[BiasAction]
        | list[list[BiasAction]],
        timestep: float,
        kT: float = 1.0,
        log: str | Path = "",
        *,
        restart: bool = False,
    ) -> None:
        """Initialize PlumedModel.

        Args:
            model: Underlying force/energy model implementing ``ModelInterface``.
            plumed_input: PLUMED commands as:

                - A ``list[str]`` of command strings (shared across all systems).
                  When ``n_systems > 1``, ``FILE=`` paths are automatically
                  suffixed with ``.{i}`` to keep output files separate.
                - A ``list[list[str]]`` of per-system command lists.  The outer
                  list must have exactly one entry per system.
                - A ``str`` or ``Path`` pointing to a PLUMED ``.dat`` file
                  (treated as shared input, with automatic file suffixing for
                  multi-system runs).
                - A ``list[BiasAction]`` of
                  :class:`~torch_sim.plumed_builder.BiasAction` objects
                  (Pythonic builder API; converted to strings automatically).
                - A ``list[list[BiasAction]]`` of per-system action lists
                  (also converted automatically).

            timestep: MD timestep in ps. Must match the timestep passed to
                ``integrate()``.
            kT: Thermal energy in eV. Used by some PLUMED actions (e.g. WTMetaD).
                Defaults to 1.0 eV.
            log: Path to PLUMED log file. Empty string means stdout.  For
                multi-system runs the log is suffixed with ``.{i}`` before the
                file extension (e.g. ``plumed.log`` → ``plumed.0.log``).
                Defaults to ``""``.
            restart: Whether to restart PLUMED from a previous run.
                Defaults to ``False``.
        """
        super().__init__()
        self.model = model
        self.plumed_input = _normalize_plumed_input(plumed_input)
        self.timestep = timestep
        self.kT = kT
        self.log = log
        self.restart = restart
        # One plumed.Plumed() instance per system; created lazily on first forward().
        self._plumed_instances: list[Any] = []
        self._step: int = 0

    # --- ModelInterface property delegation ---

    @property
    def device(self) -> torch.device:
        """Device delegated from the wrapped model."""
        return self.model.device

    @property
    def dtype(self) -> torch.dtype:
        """Data type delegated from the wrapped model."""
        return self.model.dtype

    @property
    def compute_stress(self) -> bool:
        """Stress computation flag delegated from the wrapped model."""
        return self.model.compute_stress

    @property
    def compute_forces(self) -> bool:
        """Force computation flag delegated from the wrapped model."""
        return self.model.compute_forces

    @property
    def memory_scales_with(self) -> MemoryScaling:
        """Memory scaling delegated from the wrapped model."""
        return self.model.memory_scales_with

    # --- Step counter ---

    @property
    def step(self) -> int:
        """Current PLUMED step counter.

        Can be set to synchronize the counter when restarting a simulation.
        """
        return self._step

    @step.setter
    def step(self, value: int) -> None:
        """Set the PLUMED step counter."""
        self._step = value

    # --- Internals ---

    def _input_lines_for_system(self, i: int, n_systems: int) -> list[str]:
        """Return the PLUMED command lines for system *i*.

        For per-system input (``list[list[str]]``) the *i*-th sub-list is
        returned unchanged.  For shared input (``list[str]`` or a file path),
        ``FILE=`` arguments are suffixed with ``.{i}`` when ``n_systems > 1``.

        Args:
            i: System index.
            n_systems: Total number of systems in the current state.

        Returns:
            list[str]: PLUMED command strings for system *i*.

        Raises:
            ValueError: If per-system input has fewer entries than requested.
        """
        # Per-system explicit input: each system has its own command list.
        if (
            isinstance(self.plumed_input, list)
            and self.plumed_input
            and isinstance(self.plumed_input[0], list)
        ):
            if i >= len(self.plumed_input):
                raise ValueError(
                    f"plumed_input has {len(self.plumed_input)} per-system entries "
                    f"but system index {i} was requested."
                )
            return self.plumed_input[i]  # type: ignore[return-value]

        # Shared input: load and optionally suffix FILE= args.
        lines = _load_plumed_lines(self.plumed_input)  # type: ignore[arg-type]
        if n_systems > 1:
            lines = _suffix_file_args(lines, f".{i}")
        return lines

    def _log_for_system(self, i: int, n_systems: int) -> str:
        """Return the log file path for system *i*.

        Returns an empty string (stdout) when ``self.log`` is empty.  For
        multi-system runs the base log path is suffixed with ``.{i}`` before
        the file extension (e.g. ``plumed.log`` → ``plumed.0.log``).

        Args:
            i: System index.
            n_systems: Total number of systems in the current state.

        Returns:
            str: Log file path for system *i*, or ``""`` for stdout.
        """
        if not self.log:
            return ""
        if n_systems == 1:
            return str(self.log)
        p = Path(str(self.log))
        return str(p.parent / f"{p.stem}.{i}{p.suffix}")

    def _create_plumed_instance(
        self, n_atoms: int, input_lines: list[str], log: str
    ) -> Any:
        """Create and initialize a single PLUMED instance.

        Args:
            n_atoms: Number of atoms for this system.
            input_lines: PLUMED command strings.
            log: Log file path, or empty string for stdout.

        Returns:
            plumed.Plumed: Initialized PLUMED instance ready to receive ``calc``
            commands.
        """
        import plumed

        p = plumed.Plumed()
        if log:
            p.cmd("setLogFile", log)
        p.cmd("setNatoms", n_atoms)
        p.cmd("setMDEngine", "torch-sim")
        p.cmd("setTimestep", float(self.timestep))
        p.cmd("setKbT", float(self.kT))
        p.cmd("setRestart", 1 if self.restart else 0)
        p.cmd("setMDLengthUnits", _PLUMED_LENGTH_FACTOR)
        p.cmd("setMDEnergyUnits", _PLUMED_ENERGY_FACTOR)
        p.cmd("setMDTimeUnits", _PLUMED_TIME_FACTOR)
        p.cmd("setMDMassUnits", _PLUMED_MASS_FACTOR)
        p.cmd("init")
        for line in input_lines:
            p.cmd("readInputLine", line)
        return p

    def forward(self, state: SimState, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Compute biased energies and forces.

        Calls the underlying model on the full batched state, then applies
        per-system PLUMED bias forces and energy corrections.

        Args:
            state: Simulation state.
            **kwargs: Additional keyword arguments forwarded to the underlying model.

        Returns:
            dict[str, torch.Tensor]: Computed properties with PLUMED bias applied:

                - ``"energy"``: Total energy (unbiased + bias) with shape
                  ``[n_systems]``.
                - ``"forces"``: Total forces (unbiased + bias) with shape
                  ``[n_atoms, 3]``.
                - ``"biased_energy"``: PLUMED bias energy only, with shape
                  ``[n_systems]``.
                - ``"biased_forces"``: PLUMED bias forces only, with shape
                  ``[n_atoms, 3]``.
                - ``"stress"``: Stress tensor with shape ``[n_systems, 3, 3]`` (if
                  the model computes stress; note the PLUMED virial is not applied
                  to stress).
        """
        n_systems = state.n_systems

        # Unbiased model call on the full batched state (single GPU kernel).
        model_output = self.model(state, **kwargs)

        # Grow the instance list to cover all systems (handles n_systems growth).
        while len(self._plumed_instances) < n_systems:
            self._plumed_instances.append(None)

        total_energy = model_output["energy"].clone()
        total_forces = model_output["forces"].clone()
        bias_energy = torch.zeros(n_systems, device=state.device, dtype=state.dtype)
        bias_forces = torch.zeros_like(total_forces)

        for i in range(n_systems):
            mask = state.system_idx == i
            n_atoms_i = int(mask.sum())

            # Lazy per-system PLUMED initialization.
            if self._plumed_instances[i] is None:
                self._plumed_instances[i] = self._create_plumed_instance(
                    n_atoms=n_atoms_i,
                    input_lines=self._input_lines_for_system(i, n_systems),
                    log=self._log_for_system(i, n_systems),
                )

            pos_i = state.positions[mask].detach().cpu().numpy().astype(np.float64)
            masses_i = state.masses[mask].detach().cpu().numpy().astype(np.float64)
            # TorchSim stores cell as column vectors; PLUMED expects row vectors.
            box_i = state.cell[i].T.detach().cpu().numpy().astype(np.float64)
            energy_i = float(model_output["energy"][i])

            forces_bias_i = np.zeros((n_atoms_i, 3), dtype=np.float64)
            virial_i = np.zeros((3, 3), dtype=np.float64)

            p = self._plumed_instances[i]
            p.cmd("setStep", self._step)
            p.cmd("setBox", box_i)
            p.cmd("setMasses", masses_i)
            p.cmd("setPositions", pos_i)
            p.cmd("setEnergy", energy_i)
            p.cmd("setForces", forces_bias_i)
            p.cmd("setVirial", virial_i)
            p.cmd("calc")

            bias_i = np.zeros(1, dtype=np.float64)
            p.cmd("getBias", bias_i)

            forces_bias_t = torch.from_numpy(forces_bias_i.copy()).to(
                device=state.device, dtype=state.dtype
            )
            bias_forces[mask] = forces_bias_t
            bias_energy[i] = bias_i[0]
            total_forces[mask] += forces_bias_t
            total_energy[i] += bias_i[0]

        self._step += 1

        result: dict[str, torch.Tensor] = {
            "energy": total_energy,
            "forces": total_forces,
            "biased_energy": bias_energy,
            "biased_forces": bias_forces,
        }
        if "stress" in model_output:
            result["stress"] = model_output["stress"]
        return result
