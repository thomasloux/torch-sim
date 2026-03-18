"""Thermodynamic integration module for molecular dynamics simulations."""

import os
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

import torch
from tqdm import tqdm

import torch_sim as ts
from torch_sim.autobatching import BinningAutoBatcher
from torch_sim.integrators.md import initialize_momenta, momentum_step, position_step
from torch_sim.integrators.nvt import NVTVRescaleState, _vrescale_update
from torch_sim.models.interface import ModelInterface
from torch_sim.runners import _configure_batches_iterator, _configure_reporter
from torch_sim.state import SimState, concatenate_states, initialize_state
from torch_sim.units import UnitSystem


def linear_lambda_schedule(step: int, n_steps: int) -> float:
    """Linear lambda schedule: λ(t) = t/T."""
    return step / n_steps


def quadratic_lambda_schedule(step: int, n_steps: int) -> float:
    """Quadratic lambda schedule: λ(t) = -1*(1-t/T)² + 1."""
    t_normalized = step / n_steps
    return -1 * (1 - t_normalized) ** 2 + 1


def cubic_lambda_schedule(step: int, n_steps: int) -> float:
    """Cubic lambda schedule: λ(t) = -1*(1-t/T)³ + 1."""
    t_normalized = step / n_steps
    return -1 * (1 - t_normalized) ** 3 + 1


def lammps_lambda_schedule(step: int, n_steps: int) -> float:
    """Lambda schedule used in LAMMPS paper: λ(t) = 0.5*(1 - cos(π*t/T))."""
    t = step / n_steps
    return t**5 * (70 * t**4 - 315 * t**3 + 540 * t**2 - 420 * t + 126)


LAMBDA_SCHEDULES = {
    "linear": linear_lambda_schedule,
    "quadratic": quadratic_lambda_schedule,
    "lammps": lammps_lambda_schedule,
    "cubic": cubic_lambda_schedule,
}


@dataclass(kw_only=True)
class ThermodynamicIntegrationMDState(NVTVRescaleState):
    """Custom state for thermodynamic integration in MD simulations.

    This state can hold additional properties like lambda_ for TI.
    """

    lambda_: torch.Tensor
    energy_difference: torch.Tensor
    energy1: torch.Tensor
    energy2: torch.Tensor

    _system_attributes = NVTVRescaleState._system_attributes | {  # noqa: SLF001
        "lambda_",
        "energy_difference",
        "energy1",
        "energy2",
    }


class MixedModel(ModelInterface):
    """A model that mixes two models for thermodynamic integration.

    Computes H(λ) = g(λ) H₁ + f(λ) H₂, with dH/dλ = g'(λ) H₁ + f'(λ) H₂.
    Defaults to linear mixing: f(λ)=λ, g(λ)=1-λ.
    """

    def __init__(
        self,
        model1: ModelInterface,
        model2: ModelInterface,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
        *,
        compute_stress: bool = False,
        compute_forces: bool = True,
        f: Callable[[torch.Tensor], torch.Tensor] | None = None,
        g: Callable[[torch.Tensor], torch.Tensor] | None = None,
        f_prime: Callable[[torch.Tensor], torch.Tensor] | None = None,
        g_prime: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        """Initialize the mixed model.

        Args:
            model1: First model in the mixture
            model2: Second model in the mixture
            device: Device to run computations on
            dtype: Data type for computations
            compute_stress: Whether to compute stress
            compute_forces: Whether to compute forces
            f: Mixing function for model2 energies, f(λ). Defaults to λ.
            g: Mixing function for model1 energies, g(λ). Defaults to 1-λ.
            f_prime: Derivative of f w.r.t. λ. Defaults to 1.
            g_prime: Derivative of g w.r.t. λ. Defaults to -1.
        """
        super().__init__()
        self.model1 = model1
        self.model2 = model2
        self._device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        if isinstance(self._device, str):
            self._device = torch.device(self._device)

        self._dtype = dtype
        self._compute_stress = compute_stress
        self._compute_forces = compute_forces

        # Mixing functions: H(λ) = g(λ) H₁ + f(λ) H₂
        self.f = f if f is not None else lambda x: x
        self.g = g if g is not None else lambda x: 1 - x
        self.f_prime = f_prime if f_prime is not None else lambda x: torch.ones_like(x)  # noqa: PLW0108
        self.g_prime = (
            g_prime if g_prime is not None else lambda x: -torch.ones_like(x)
        )

    def forward(self, state: ts.SimState) -> dict[str, torch.Tensor]:
        """Forward pass through the mixed model.

        Args:
            state: Simulation state containing positions, masses, etc.

        Returns:
            Dictionary with mixed energies and forces
        """
        lambda_ = state.lambda_
        lambda_per_atom = lambda_[state.system_idx]
        out1 = self.model1(state)
        out2 = self.model2(state)

        # H(λ) = g(λ) H₁ + f(λ) H₂
        g_val = self.g(lambda_)
        f_val = self.f(lambda_)
        g_val_per_atom = self.g(lambda_per_atom)
        f_val_per_atom = self.f(lambda_per_atom)

        output = {}
        output["energy"] = g_val * out1["energy"] + f_val * out2["energy"]
        output["forces"] = g_val_per_atom.view(-1, 1) * out1[
            "forces"
        ] + f_val_per_atom.view(-1, 1) * out2["forces"]

        # dH/dλ = g'(λ) H₁ + f'(λ) H₂
        output["energy_difference"] = (
            self.g_prime(lambda_) * out1["energy"]
            + self.f_prime(lambda_) * out2["energy"]
        )
        output["energy1"] = out1["energy"]
        output["energy2"] = out2["energy"]
        return output


def power_mixing(
    m: float,
) -> dict[str, Callable[[torch.Tensor], torch.Tensor]]:
    """Return f, g, f_prime, g_prime for power-law mixing with exponent m.

    Produces mixing functions f(λ) = λ^m and g(λ) = (1-λ)^m, so that
    H(λ) = (1-λ)^m H₁ + λ^m H₂.

    Args:
        m: Power-law exponent. m=1 recovers standard linear mixing.

    Returns:
        Dictionary with keys "f", "g", "f_prime", "g_prime" suitable for
        passing to MixedModel via ``MixedModel(..., **power_mixing(m))``.
    """
    return {
        "f": lambda x, _m=m: x**_m,
        "g": lambda x, _m=m: (1 - x) ** _m,
        "f_prime": lambda x, _m=m: _m * x ** (_m - 1),
        "g_prime": lambda x, _m=m: -_m * (1 - x) ** (_m - 1),
    }


def velocity_verlet_steered_md[T: ThermodynamicIntegrationMDState](
    state: T, dt: float | torch.Tensor, model: ModelInterface
) -> T:
    """Perform one complete velocity Verlet integration step.

    This function implements the velocity Verlet algorithm, which provides
    time-reversible integration of the equations of motion. The integration
    sequence is:
    1. Half momentum update
    2. Full position update
    3. Force update
    4. Half momentum update

    Args:
        state: Current system state containing positions, momenta, forces
        dt: Integration timestep
        model: Neural network model that computes energies and forces

    Returns:
        Updated state after one complete velocity Verlet step

    Notes:
        - Time-reversible and symplectic integrator of second order accuracy
        - Conserves energy in the absence of numerical errors
        - Handles periodic boundary conditions if enabled in state
    """
    dt = torch.as_tensor(dt, device=state.device, dtype=state.dtype)
    dt_2 = dt / 2
    state = momentum_step(state, dt_2)
    state = position_step(state, dt)

    model_output = model(state)

    state.energy = model_output["energy"]
    state.forces = model_output["forces"]
    state.energy_difference = model_output["energy_difference"]
    state.energy1 = model_output["energy1"]
    state.energy2 = model_output["energy2"]
    return momentum_step(state, dt_2)


def nvt_vrescale_steered_init(
    state: SimState,
    model: ModelInterface,
    *,
    kT: float | torch.Tensor,
    lambda_: torch.Tensor,
    **_kwargs: Any,
) -> ThermodynamicIntegrationMDState:
    """Initialize an NVT state from input data for velocity rescaling dynamics.

    Creates an initial state for NVT molecular dynamics using the canonical
    sampling through velocity rescaling (CSVR) thermostat. This thermostat
    samples from the canonical ensemble by rescaling velocities with an
    appropriately chosen random factor.

    To seed the RNG set ``state.rng = seed`` before calling.

    Args:
        model: Neural network model that computes energies and forces.
            Must return a dict with 'energy' and 'forces' keys.
        state: SimState containing positions, masses, cell, pbc, and other
            required state vars
        kT: Temperature in energy units for initializing momenta,
            either scalar or with shape [n_systems]
        lambda_: Lambda values for thermodynamic integration, shape [n_systems]

    Returns:
        ThermodynamicIntegrationMDState: Initialized state for NVT integration
            containing positions, momenta, forces, energy, and other required
            attributes

    Notes:
        The initial momenta are sampled from a Maxwell-Boltzmann distribution
        at the specified temperature. The V-Rescale thermostat provides proper
        canonical sampling through stochastic velocity rescaling.
    """
    state.lambda_ = lambda_
    model_output = model(state)

    momenta = getattr(state, "momenta", None)
    if momenta is None:
        momenta = initialize_momenta(
            state.positions,
            state.masses,
            state.system_idx,
            kT,
            state.rng,
        )

    return ThermodynamicIntegrationMDState.from_state(
        state,
        momenta=momenta,
        energy=model_output["energy"],
        forces=model_output["forces"],
        energy_difference=model_output["energy_difference"],
        energy1=model_output["energy1"],
        energy2=model_output["energy2"],
        lambda_=lambda_,
    )


def nvt_vrescale_steered_step(
    model: ModelInterface,
    state: ThermodynamicIntegrationMDState,
    *,
    dt: float | torch.Tensor,
    kT: float | torch.Tensor,
    tau: float | torch.Tensor | None = None,
) -> ThermodynamicIntegrationMDState:
    """Perform one complete V-Rescale dynamics integration step.

    This function implements the canonical sampling through velocity rescaling (V-Rescale)
    thermostat combined with velocity Verlet integration. The V-Rescale thermostat samples
    the canonical distribution by rescaling velocities with a properly chosen random
    factor that ensures correct canonical sampling.

    Args:
        model: Neural network model that computes energies and forces.
            Must return a dict with 'energy' and 'forces' keys.
        state: Current system state containing positions, momenta, forces
        dt: Integration timestep, either scalar or shape [n_systems]
        kT: Target temperature in energy units, either scalar or
            with shape [n_systems]
        tau: Thermostat relaxation time controlling the coupling strength,
            either scalar or with shape [n_systems]. Defaults to 100*dt.

    Returns:
        ThermodynamicIntegrationMDState: Updated state after one complete
            V-Rescale step with new positions, momenta, forces, and energy

    Notes:
        - Uses V-Rescale thermostat for proper canonical ensemble sampling
        - Unlike Berendsen thermostat, V-Rescale samples the true canonical distribution
        - Integration sequence: V-Rescale rescaling + Velocity Verlet step
        - The rescaling factor follows the distribution derived in Bussi et al.

    References:
        Bussi G, Donadio D, Parrinello M. "Canonical sampling through velocity rescaling."
        The Journal of chemical physics, 126(1), 014101 (2007).
    """
    device, dtype = model.device, model.dtype

    tau = torch.as_tensor(100 * dt if tau is None else tau, device=device, dtype=dtype)
    dt = torch.as_tensor(dt, device=device, dtype=dtype)
    kT = torch.as_tensor(kT, device=device, dtype=dtype)

    # Apply V-Rescale rescaling
    state = _vrescale_update(state, tau, kT, dt)

    # Perform velocity Verlet step
    return velocity_verlet_steered_md(state=state, dt=dt, model=model)


def run_non_equilibrium_md(  # noqa: C901 PLR0915
    system: Any,
    model_a: ModelInterface,
    model_b: ModelInterface,
    save_dir: str,
    *,
    n_steps: int = 1000,
    lambda_schedule: str | Callable = "linear",
    reverse: bool = False,
    temperature: float = 300.0,
    timestep: float = 0.002,
    pbar: bool | dict[str, Any] = False,
    trajectory_reporter: ts.TrajectoryReporter | None = None,
    step_frequency: int = 1,
    autobatcher: bool = False,
    state_frequency: int = 50,
    mixing_functions: dict[str, Callable] | None = None,
) -> ts.SimState:
    """Run non-equilibrium molecular dynamics simulation.

    Args:
        system: Initial system state, possibly batched
        model_a: First model for thermodynamic integration
        model_b: Second model for thermodynamic integration
        save_dir: Directory to save trajectory files
        n_steps: Number of simulation steps
        lambda_schedule: Lambda schedule type ("linear", "quadratic", "paper")
        reverse: Reverse the Lambda schedule for backward TI
            for non symmetric lambda paths
        temperature: Temperature for simulation
        timestep: Integration timestep
        pbar (bool | dict[str, Any], optional): Show a progress bar.
            Only works with an autobatcher in interactive shell. If a dict is given,
            it's passed to `tqdm` as kwargs.
        trajectory_reporter: Reporter for trajectory data
        step_frequency: Frequency for reporting steps
        autobatcher: Whether to use automatic batching
        state_frequency: Frequency for state reporting
        mixing_functions: Optional dict with keys "f", "g", "f_prime", "g_prime"
            for custom mixing. See :func:`power_mixing` for a convenience factory.

    Returns:
        Final simulation state
    """
    unit_system = UnitSystem.metal

    # Validate lambda schedule
    if isinstance(lambda_schedule, str):
        if lambda_schedule not in LAMBDA_SCHEDULES:
            raise ValueError(
                f"Unknown lambda schedule: {lambda_schedule}. "
                f"Available: {list(LAMBDA_SCHEDULES.keys())}"
            )
        schedule_fn = LAMBDA_SCHEDULES[lambda_schedule]

    if isinstance(lambda_schedule, Callable):
        schedule_fn = lambda_schedule

    def lambda_schedule_fn(step: int) -> float:
        if reverse:
            return schedule_fn(n_steps - 1 - step, n_steps - 1)
        return schedule_fn(step, n_steps - 1)

    # Ensure system is a single system (not batched)
    if isinstance(system, list):
        raise TypeError("system should be a single system, not a list. ")

    model = MixedModel(
        model1=model_a,
        model2=model_b,
        device=model_b.device,
        dtype=model_b.dtype,
        **(mixing_functions or {}),
    )
    state: SimState = initialize_state(system, model.device, model.dtype)
    dtype, device = state.dtype, state.device
    kT = (
        torch.as_tensor(temperature, dtype=dtype, device=device) * unit_system.temperature
    )
    dt = torch.as_tensor(timestep * UnitSystem.metal.time, dtype=dtype, device=device)

    # Create filenames for trajectory files
    filenames = [
        os.path.join(save_dir, f"trajectory_steered_{replica_idx}.h5")
        for replica_idx in range(state.n_systems)
    ]

    trajectory_reporter = ts.TrajectoryReporter(
        filenames=filenames,
        state_frequency=state_frequency,
        prop_calculators={
            step_frequency: {
                "energy_diff": lambda state: state.energy_difference,
                "energy": lambda state: state.energy,
                "energy1": lambda state: state.energy1,
                "energy2": lambda state: state.energy2,
                "lambda_": lambda state: state.lambda_,
            },
            10: {
                "temperature": lambda state: ts.quantities.calc_temperature(
                    masses=state.masses,
                    momenta=state.momenta,
                    system_idx=state.system_idx,
                )
            },
        },
    )

    if not kT.ndim == 0:
        raise TypeError("temperature must be a single float value.")

    # This can be modified to accept any integrator compatible with
    # ThermodynamicIntegrationMDState
    init_fn, update_fn = nvt_vrescale_steered_init, nvt_vrescale_steered_step
    init_fn = partial(init_fn, model=model)
    update_fn = partial(update_fn, model, dt=dt)

    # batch_iterator will be a list if autobatcher is False
    batch_iterator = _configure_batches_iterator(state, model, autobatcher=autobatcher)
    trajectory_reporter = _configure_reporter(
        trajectory_reporter,
        properties=["kinetic_energy", "potential_energy", "temperature"],
    )

    final_states: list[SimState] = []

    tqdm_pbar = None
    if pbar and autobatcher:
        pbar_kwargs = pbar if isinstance(pbar, dict) else {}
        pbar_kwargs.setdefault("desc", "Integrate")
        pbar_kwargs.setdefault("disable", None)
        tqdm_pbar = tqdm(total=state.n_systems, **pbar_kwargs)

    for state, system_indices in batch_iterator:
        # Initialize lambda values based on batch indices
        lambda_values = torch.ones(
            state.n_systems, dtype=dtype, device=device
        ) * lambda_schedule_fn(0)
        state = init_fn(state=state, lambda_=lambda_values, kT=kT)

        # set up trajectory reporters
        if autobatcher and trajectory_reporter:
            # we must remake the trajectory reporter for each batch
            trajectory_reporter.reopen_trajectories(
                filenames=[filenames[i] for i in system_indices]
            )

        # Thermodynamic integration phase
        ti_bar = tqdm(
            range(1, n_steps + 1),
            desc="TI Integration",
            disable=not pbar,
            mininterval=0.5,
        )

        for step in ti_bar:
            # Calculate lambda using the selected schedule
            lambda_value = lambda_schedule_fn(step - 1)

            # Update lambda values
            new_lambdas = torch.full(
                (state.n_systems,), lambda_value, dtype=dtype, device=device
            )

            state.lambda_ = new_lambdas

            # Update state
            state = update_fn(state=state, kT=kT)

            if trajectory_reporter:
                trajectory_reporter.report(state, step, model=model)

        # finish the trajectory reporter
        final_states.append(state)
        if tqdm_pbar:
            tqdm_pbar.update(state.n_systems)

    if trajectory_reporter:
        trajectory_reporter.finish()

    if isinstance(batch_iterator, BinningAutoBatcher):
        reordered_states = batch_iterator.restore_original_order(final_states)
        return concatenate_states(reordered_states)

    return state


def run_equilibrium_md(  # noqa: C901
    system: Any,
    model_a: ModelInterface,
    model_b: ModelInterface,
    lambdas: torch.Tensor,
    save_dir: str,
    *,
    n_steps: int = 1000,
    temperature: float = 300.0,
    timestep: float = 0.002,
    pbar: bool | dict[str, Any] = False,
    trajectory_reporter: ts.TrajectoryReporter | None = None,
    step_frequency: int = 1,
    filenames: str | None = None,
    autobatcher: bool = False,
    state_frequency: int = 50,
    mixing_functions: dict[str, Callable] | None = None,
) -> ts.SimState:
    """Run equilibrium molecular dynamics simulation.

    Args:
        system: Initial system state, possibly batched
        model_a: First model for thermodynamic integration
        model_b: Second model for thermodynamic integration
        lambdas: Tensor of lambda values for each system in the batch
        save_dir: Directory to save trajectory files
        n_steps: Number of simulation steps
        temperature: Temperature for simulation
        timestep: Integration timestep
        pbar (bool | dict[str, Any], optional): Show a progress bar.
            Only works with an autobatcher in interactive shell. If a dict is given,
            it's passed to `tqdm` as kwargs.
        trajectory_reporter: Reporter for trajectory data
        step_frequency: Frequency for reporting steps
        filenames: List of filenames for trajectory files. If None, defaults will be used.
            Useful when running sequential thermodynamic integration
        autobatcher: Whether to use automatic batching
        state_frequency: Frequency for state reporting
        mixing_functions: Optional dict with keys "f", "g", "f_prime", "g_prime"
            for custom mixing. See :func:`power_mixing` for a convenience factory.

    Returns:
        Final simulation state
    """
    unit_system = UnitSystem.metal

    if lambdas.ndim == 0:
        lambdas = lambdas.unsqueeze(0)

    # Ensure system is a single system (not batched)
    if isinstance(system, list):
        raise TypeError("system should be a single system, not a list. ")
    if len(lambdas) != len(lambdas.unique()):
        raise ValueError(
            "Lambda list must be unique.Batch of different systems is not supported yet."
        )

    model = MixedModel(
        model1=model_a,
        model2=model_b,
        device=model_b.device,
        dtype=model_b.dtype,
        **(mixing_functions or {}),
    )
    state: SimState = initialize_state(system, model.device, model.dtype)
    dtype, device = state.dtype, state.device
    kT = (
        torch.as_tensor(temperature, dtype=dtype, device=device) * unit_system.temperature
    )
    dt = torch.as_tensor(timestep * UnitSystem.metal.time, dtype=dtype, device=device)
    if state.n_systems != len(lambdas):
        raise ValueError(
            f"Number of systems in state ({state.n_systems}) must match "
            f"number of lambda values ({len(lambdas)})."
        )

    # Create filenames for trajectory files
    if filenames is None:
        filenames = [
            os.path.join(save_dir, f"trajectory_lambda_{replica_idx}.h5")
            for replica_idx in range(len(lambdas))
        ]
    else:
        filenames = [os.path.join(save_dir, filename) for filename in filenames]

    trajectory_reporter = ts.TrajectoryReporter(
        filenames=filenames,
        state_frequency=state_frequency,
        prop_calculators={
            step_frequency: {
                "energy_diff": lambda state: state.energy_difference,
                "energy": lambda state: state.energy,
                # "energy1": lambda state: state.energy1,
                # "energy2": lambda state: state.energy2,
                "lambda_": lambda state: state.lambda_,
            },
            10: {
                "temperature": lambda state: ts.quantities.calc_temperature(
                    masses=state.masses,
                    momenta=state.momenta,
                    system_idx=state.system_idx,
                )
            },
        },
    )

    if not kT.ndim == 0:
        raise TypeError("temperature must be a single float value.")

    init_fn, update_fn = nvt_vrescale_steered_init, nvt_vrescale_steered_step
    init_fn = partial(init_fn, model=model)
    update_fn = partial(update_fn, model, dt=dt)

    # batch_iterator will be a list if autobatcher is False
    batch_iterator = _configure_batches_iterator(state, model, autobatcher=autobatcher)
    trajectory_reporter = _configure_reporter(
        trajectory_reporter,
        properties=["kinetic_energy", "potential_energy", "temperature"],
    )

    final_states: list[SimState] = []

    tqdm_pbar = None
    if pbar and autobatcher:
        pbar_kwargs = pbar if isinstance(pbar, dict) else {}
        pbar_kwargs.setdefault("desc", "Integrate")
        pbar_kwargs.setdefault("disable", None)
        tqdm_pbar = tqdm(total=state.n_systems, **pbar_kwargs)

    for state, system_indices in batch_iterator:
        batch_lambdas = lambdas[system_indices] if system_indices else lambdas
        state = init_fn(state=state, lambda_=batch_lambdas, kT=kT)

        # set up trajectory reporters
        if autobatcher and trajectory_reporter:
            # we must remake the trajectory reporter for each batch
            trajectory_reporter.reopen_trajectories(
                filenames=[filenames[i] for i in system_indices]
            )

        # Thermodynamic integration phase
        ti_bar = tqdm(
            range(1, n_steps + 1),
            desc="TI Integration",
            disable=not pbar,
            mininterval=0.5,
        )

        for step in ti_bar:
            # Update state
            state = update_fn(state=state, kT=kT)

            if trajectory_reporter:
                trajectory_reporter.report(state, step, model=model)

        # finish the trajectory reporter
        final_states.append(state)
        if tqdm_pbar:
            tqdm_pbar.update(state.n_systems)

    if trajectory_reporter:
        trajectory_reporter.finish()

    if isinstance(batch_iterator, BinningAutoBatcher):
        reordered_states = batch_iterator.restore_original_order(final_states)
        return concatenate_states(reordered_states)

    return state
