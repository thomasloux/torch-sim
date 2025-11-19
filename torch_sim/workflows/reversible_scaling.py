"""Reversible Scaling MD workflow for free energy calculations."""

import logging
import os
from dataclasses import dataclass
from functools import partial

import torch
from tqdm import tqdm

import torch_sim as ts
from torch_sim.autobatching import BinningAutoBatcher
from torch_sim.integrators import INTEGRATOR_REGISTRY
from torch_sim.integrators.md import momentum_step
from torch_sim.integrators.npt import (
    NPTCRescaleState,
    _crescale_isotropic_barostat_step,
    _vrescale_update,
    npt_crescale_init,
)
from torch_sim.models.interface import ModelInterface
from torch_sim.runners import _configure_batches_iterator, _configure_reporter
from torch_sim.state import SimState, concatenate_states, initialize_state
from torch_sim.typing import StateDict
from torch_sim.units import UnitSystem


# from torch_sim.workflows.free_energy_prediction import (
#     compute_free_energy_jarzynski,
# )

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class ReversibleScalingMDState(NPTCRescaleState):
    """Custom state for reversible scaling in MD simulations for free energy calculations.

    This state holds additional properties: lambda_ and unscaled_energy, which are
    essential for performing reversible scaling thermodynamic integration.
    """

    lambda_: torch.Tensor
    unscaled_energy: torch.Tensor

    _system_attributes = NPTCRescaleState._system_attributes | {  # noqa: SLF001
        "lambda_",
        "unscaled_energy",
    }


class ScaledModel(ModelInterface):
    """A model that scales the potential energy by a lambda factor."""

    def __init__(self, model: ModelInterface) -> None:
        """Initialize the ScaledModel with a base model."""
        super().__init__()
        self.model = model
        self._device = model.device
        self._dtype = model.dtype
        self._compute_stress = model.compute_stress
        self._compute_forces = model.compute_forces

    def forward(self, state: ReversibleScalingMDState) -> dict[str, torch.Tensor]:
        """Forward pass that scales energy, forces, and stress by lambda."""
        results = self.model.forward(state)
        results["unscaled_energy"] = results["energy"]
        results["energy"] = results["energy"] * state.lambda_
        results["forces"] = results["forces"] * state.lambda_[state.system_idx][:, None]
        results["stress"] = results["stress"] * state.lambda_[:, None, None]
        return results


def npt_crescale_rs_init(
    state: SimState | StateDict,
    model: ModelInterface,
    *,
    kT: torch.Tensor,
    dt: torch.Tensor,
    tau_p: torch.Tensor | None = None,
    isothermal_compressibility: torch.Tensor | None = None,
    seed: int | None = None,
) -> ReversibleScalingMDState:
    """Initialize the NPT cell rescaling reversible scaling MD state."""
    state.lambda_ = torch.ones(state.n_systems, device=state.device, dtype=state.dtype)
    state = npt_crescale_init(
        state=state,
        model=model,
        kT=kT,
        dt=dt,
        tau_p=tau_p,
        isothermal_compressibility=isothermal_compressibility,
        seed=seed,
    )

    return ReversibleScalingMDState(
        # **asdict(state),
        positions=state.positions,
        masses=state.masses,
        cell=state.cell,
        pbc=state.pbc,
        atomic_numbers=state.atomic_numbers,
        system_idx=state.system_idx,
        momenta=state.momenta,
        energy=state.energy,
        forces=state.forces,
        stress=state.stress,
        isothermal_compressibility=isothermal_compressibility,
        tau_p=tau_p,
        lambda_=torch.ones(state.n_systems, device=state.device, dtype=state.dtype),
        unscaled_energy=state.energy,
    )


def npt_crescale_isotropic_rs_step(
    state: ReversibleScalingMDState,
    model: ModelInterface,
    *,
    dt: torch.Tensor,
    kT: torch.Tensor,
    external_pressure: torch.Tensor,
    tau: torch.Tensor | None = None,
) -> ReversibleScalingMDState:
    """Perform one NPT integration step with cell rescaling barostat.

    See npt_crescale_isotropic_barostat_step for details on the barostat step.
    Update necessary properties in the state.

    Args:
        model (ModelInterface): Model to compute forces and energies
        state (ReversibleScalingMDState): Current system state
        dt (torch.Tensor): Integration timestep
        kT (torch.Tensor): Target temperature
        external_pressure (torch.Tensor): Target external pressure
        tau (torch.Tensor | None): V-Rescale thermostat relaxation time. If None,
            defaults to 100*dt

    Returns:
        ReversibleScalingMDState: Updated state after one integration step
    """
    # Note: would probably be better to have tau in NVTCRescaleState
    if tau is None:
        tau = 100 * dt
    state = _vrescale_update(state, tau, kT, dt / 2)

    state = momentum_step(state, dt / 2)

    # Barostat step
    state = _crescale_isotropic_barostat_step(state, kT, dt, external_pressure)

    # Forces
    model_output = model(state)
    state.forces = model_output["forces"]
    state.energy = model_output["energy"]
    state.stress = model_output["stress"]
    state.unscaled_energy = model_output["unscaled_energy"]

    # Final momentum step
    state = momentum_step(state, dt / 2)

    # Final thermostat step
    return _vrescale_update(state, tau, kT, dt / 2)


def lambda_schedule_fn(step: int, final_lambda: float, n_steps: int) -> float:
    """Compute lambda value at a given step using a linear schedule of temperature."""
    return 1 / (1 + step / (n_steps - 1) * (1 / final_lambda - 1))


def run_reversible_scaling_steered_md(  # noqa: C901, PLR0915
    system: SimState,
    model: ModelInterface,
    temperature_start: float,
    temperature_end: float,
    external_pressure: float,
    save_dir: str,
    *,
    backward: bool = False,
    tau: torch.Tensor | None = None,
    tau_p: torch.Tensor | None = None,
    isothermal_compressibility: torch.Tensor | None = None,
    n_steps: int = 1000,
    timestep: float = 0.002,
    pbar: bool = True,
    autobatcher: bool = False,
    step_frequency: int = 1,
    state_frequency: int = 50,
) -> ReversibleScalingMDState:
    """Run reversible scaling steered MD simulation.

    Args:
        system: Initial system state.
        model: Target model.
        temperature_start: Starting temperature and simulation temperature in K.
        temperature_end: Ending temperature in K.
        external_pressure: External pressure in bar.
        save_dir: Directory to save trajectory files.
        backward: Whether to run the simulation backward
            (from temperature_end to temperature_start).
        tau: Thermostat relaxation time (MetalUnits).
        tau_p: Barostat relaxation time (MetalUnits).
        isothermal_compressibility: Isothermal compressibility (MetalUnits).
        n_steps: Number of TI steps.
        timestep: Integration timestep.
        pbar: Whether to show a progress bar.
        autobatcher: Whether to use autobatching.
        step_frequency: Frequency to record properties in trajectory.
        state_frequency: Frequency to save states in trajectory.

    Returns:
        Final system state after reversible scaling MD.
    """
    os.makedirs(save_dir, exist_ok=True)

    temperature = temperature_start
    lambda_final = temperature_start / temperature_end

    # Ensure system is a single system (not batched)
    if isinstance(system, list):
        raise TypeError("system should be a single system, not a list. ")

    state: SimState = initialize_state(system, model.device, model.dtype)
    dtype, device = state.dtype, state.device
    kT = (
        torch.as_tensor(temperature, dtype=dtype, device=device)
        * UnitSystem.metal.temperature
    )
    external_pressure: torch.Tensor = (
        torch.as_tensor(external_pressure, dtype=dtype, device=device)
        * UnitSystem.metal.pressure
    )
    dt = torch.tensor(timestep * UnitSystem.metal.time, dtype=dtype, device=device)

    # Create filenames for trajectory files
    filenames = [
        os.path.join(save_dir, f"trajectory_reversible_scaling_{i}.h5")
        for i in range(system.n_systems)
    ]

    scaled_model = ScaledModel(model)

    trajectory_reporter = ts.TrajectoryReporter(
        filenames=filenames,
        state_frequency=state_frequency,
        prop_calculators={
            step_frequency: {
                "energy": lambda state: state.energy,
                "unscaled_energy": lambda state: state.unscaled_energy,
                "lambda_": lambda state: state.lambda_,
                "volume": lambda state: torch.det(state.cell),
            },
            100: {
                "temperature": lambda state: ts.quantities.calc_temperature(
                    masses=state.masses,
                    momenta=state.momenta,
                    system_idx=state.system_idx,
                ),
                "pressure": lambda state: ts.quantities.get_pressure(
                    stress=state.stress,
                    kinetic_energy=ts.quantities.calc_kinetic_energy(
                        masses=state.masses,
                        momenta=state.momenta,
                        system_idx=state.system_idx,
                    ),
                    volume=torch.det(state.cell),
                ),
            },
        },
    )

    if not kT.ndim == 0:
        raise TypeError("temperature must be a single float value.")

    # This can be modified to accept any integrator compatible with
    # ThermodynamicIntegrationMDState
    init_fn, update_fn = npt_crescale_rs_init, npt_crescale_isotropic_rs_step
    init_fn = partial(
        init_fn,
        model=scaled_model,
        tau_p=tau_p,
        isothermal_compressibility=isothermal_compressibility,
        dt=dt,
    )
    update_fn = partial(
        update_fn, dt=dt, model=scaled_model, external_pressure=external_pressure, tau=tau
    )
    if backward:
        # Modify lambda schedule for backward direction
        def lambda_schedule(step: int) -> float:
            return lambda_schedule_fn(
                n_steps - step - 1, final_lambda=lambda_final, n_steps=n_steps
            )
    else:
        lambda_schedule = partial(
            lambda_schedule_fn, final_lambda=lambda_final, n_steps=n_steps
        )

    # batch_iterator will be a list if autobatcher is False
    batch_iterator = _configure_batches_iterator(
        state, scaled_model, autobatcher=autobatcher
    )
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

    for state, batch_indices in batch_iterator:
        state = init_fn(state=state, kT=kT)

        # set up trajectory reporters
        if autobatcher and trajectory_reporter:
            # we must remake the trajectory reporter for each batch
            trajectory_reporter.load_new_trajectories(
                filenames=[filenames[i] for i in batch_indices]
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
            lambda_value = lambda_schedule(step - 1)

            # Update lambda values
            if len(batch_indices) > 0:
                new_lambdas = torch.full_like(
                    batch_indices, lambda_value, dtype=dtype, device=device
                )
            else:
                new_lambdas = torch.full(
                    (state.n_systems,), lambda_value, dtype=dtype, device=device
                )

            state.lambda_ = new_lambdas

            # Update state
            state = update_fn(state=state, kT=kT)

            if trajectory_reporter:
                trajectory_reporter.report(state, step, model=scaled_model)

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


def compute_work_reversible_scaling(
    energy_per_atom: torch.Tensor,
    lambda_vals: torch.Tensor,
) -> torch.Tensor:
    """Compute work from reversible scaling steered MD.

    Args:
        energy_per_atom: Energy per atom at each step.
        lambda_vals: Lambda values at each step.

    Returns:
        Work values for each trajectory.
    """
    delta_lambda = lambda_vals[1:] - lambda_vals[:-1]
    return torch.cumsum(energy_per_atom[:-1] * delta_lambda, dim=-1)


def run_forward_backward_reversible_scaling(
    system: SimState,
    model: ModelInterface,
    temperature_start: float,
    temperature_end: float,
    external_pressure: float,
    save_dir: str,
    free_energy_initial_temperature: float,
    *,
    tau: float | None = None,
    tau_p: float | None = None,
    isothermal_compressibility: float | None = None,
    n_trajectories: int = 10,
    n_rs_steps: int = 1000,
    n_equil_steps: int = 500,
    timestep: float = 0.002,
) -> dict[str, torch.Tensor]:
    """Run forward-backward thermodynamic integration workflow.

    Workflow:
    1. Run forward Reversible Scaling from (T_0 to T_1)
    2. Equilibrate at model for n_equil_steps
    3. Run backward Reversible Scaling from model (T_1 to T_0)
    4. Compute free energy using linear response theory

    Args:
        system: Initial system state.
        model: Target model.
        temperature_start: Starting temperature in K.
        temperature_end: Final temperature in K.
        external_pressure: External pressure in bar.
        save_dir: Directory to save trajectory files.
        free_energy_initial_temperature: Free energy at initial temperature T_0.
        tau: Thermostat relaxation time (MetalUnits).
        tau_p: Barostat relaxation time (MetalUnits).
        isothermal_compressibility: Isothermal compressibility (MetalUnits).
        n_trajectories: Number of TI trajectories.
        n_rs_steps: Number of TI steps per trajectory.
        n_equil_steps: Number of equilibration steps at model_b.
        timestep: Integration timestep.

    Returns:
        Dictionary with free energy results and trajectory data.
    """
    os.makedirs(save_dir, exist_ok=True)

    if system.n_systems != 1:
        raise NotImplementedError("Only single system input is supported.")

    # Prepare batched systems for multiple trajectories
    systems = [system.clone() for _ in range(n_trajectories)]
    batched_system = ts.state.concatenate_states(systems)

    logger.info("Running %d forward TI trajectories...", n_trajectories)

    # Create separate save directories for forward and backward
    forward_dir = os.path.join(save_dir, "forward")
    backward_dir = os.path.join(save_dir, "backward")
    os.makedirs(forward_dir, exist_ok=True)
    os.makedirs(backward_dir, exist_ok=True)

    device, dtype = batched_system.device, batched_system.dtype
    tau: torch.Tensor | None = (
        torch.as_tensor(tau, dtype=dtype, device=device)
        .unsqueeze(0)
        .repeat(n_trajectories)
        if tau is not None
        else None
    )
    tau_p: torch.Tensor | None = (
        torch.as_tensor(tau_p, dtype=dtype, device=device)
        .unsqueeze(0)
        .repeat(n_trajectories)
        if tau_p is not None
        else None
    )
    isothermal_compressibility: torch.Tensor | None = (
        torch.as_tensor(isothermal_compressibility, dtype=dtype, device=device)
        .unsqueeze(0)
        .repeat(n_trajectories)
        if isothermal_compressibility is not None
        else None
    )

    init_fn, step_fn = INTEGRATOR_REGISTRY[ts.Integrator.npt_isotropic_crescale]
    init_fn = partial(
        init_fn,
        tau_p=tau_p,
        isothermal_compressibility=isothermal_compressibility,
    )
    step_fn = partial(
        step_fn,
        tau=tau,
        external_pressure=torch.as_tensor(external_pressure, dtype=dtype, device=device)
        * ts.units.MetalUnits.pressure,
    )
    # Equilibrate at model_b
    logger.info("Equilibrating at target model for %d steps...", n_equil_steps)
    equilibrated_state = ts.integrate(
        system=batched_system,
        model=model,
        integrator=(init_fn, step_fn),
        n_steps=n_equil_steps,
        temperature=temperature_start,
        timestep=timestep,
    )

    # Run forward TI (A -> B)
    forward_final_state = run_reversible_scaling_steered_md(
        system=equilibrated_state,
        model=model,
        temperature_start=temperature_start,
        temperature_end=temperature_end,
        external_pressure=external_pressure,
        save_dir=forward_dir,
        n_steps=n_rs_steps,
        timestep=timestep,
        pbar=True,
        tau=tau,
        tau_p=tau_p,
        isothermal_compressibility=isothermal_compressibility,
    )

    # # Equilibrate at model_b
    # logger.info("Equilibrating at target model for %d steps...", n_equil_steps)
    # equilibrated_state = ts.integrate(
    #     system=forward_final_state,
    #     model=model,
    #     integrator=(init_fn, step_fn),
    #     n_steps=n_equil_steps,
    #     temperature=temperature_end,
    #     timestep=timestep,
    # )

    logger.info("Running %d backward TI trajectories...", n_trajectories)

    # Run backward TI (B -> A)
    _ = run_reversible_scaling_steered_md(
        system=forward_final_state,
        model=model,
        temperature_start=temperature_start,
        temperature_end=temperature_end,
        external_pressure=external_pressure,
        save_dir=backward_dir,
        backward=True,
        n_steps=n_rs_steps,
        timestep=timestep,
        pbar=True,
        tau=tau,
        tau_p=tau_p,
        isothermal_compressibility=isothermal_compressibility,
    )

    # Compute work values for both directions
    forward_work_values = []
    backward_work_values = []

    external_pressure: torch.Tensor = (
        torch.as_tensor(external_pressure, dtype=dtype, device=device)
        * ts.units.MetalUnits.pressure
    )
    for i in range(n_trajectories):
        # Forward work
        forward_filename = os.path.join(
            forward_dir, f"trajectory_reversible_scaling_{i}.h5"
        )
        with ts.TorchSimTrajectory(forward_filename, mode="r") as traj:
            unscaled_energies = (
                torch.from_numpy(traj.get_array("unscaled_energy"))
                .to(batched_system.device)
                .squeeze()
            )
            lambda_vals = (
                torch.from_numpy(traj.get_array("lambda_"))
                .to(batched_system.device)
                .squeeze()
            )
            volumes = (
                torch.from_numpy(traj.get_array("volume"))
                .to(batched_system.device)
                .squeeze()
            )
            unscaled_energies += external_pressure * volumes

            energy_per_atom = unscaled_energies / batched_system.n_atoms_per_system[i]

            forward_work = compute_work_reversible_scaling(energy_per_atom, lambda_vals)
            forward_work_values.append(forward_work)

        # Backward work
        backward_filename = os.path.join(
            backward_dir, f"trajectory_reversible_scaling_{i}.h5"
        )
        with ts.TorchSimTrajectory(backward_filename, mode="r") as traj:
            unscaled_energies = (
                torch.from_numpy(traj.get_array("unscaled_energy"))
                .to(batched_system.device)
                .squeeze()
            )
            lambda_vals = (
                torch.from_numpy(traj.get_array("lambda_"))
                .to(batched_system.device)
                .squeeze()
            )
            volumes = (
                torch.from_numpy(traj.get_array("volume"))
                .to(batched_system.device)
                .squeeze()
            )
            unscaled_energies += external_pressure * volumes

            energy_per_atom = unscaled_energies / batched_system.n_atoms_per_system[i]

            backward_work = -compute_work_reversible_scaling(
                energy_per_atom.flip(0), lambda_vals.flip(0)
            )
            backward_work_values.append(backward_work)

    forward_work = torch.stack(forward_work_values).to(batched_system.device)
    backward_work = torch.stack(backward_work_values).to(batched_system.device)

    # forward_free_energy = compute_free_energy_jarzynski(forward_work, temperature)
    # backward_free_energy = -compute_free_energy_jarzynski(backward_work, temperature)

    free_energy_difference = forward_work.mean(dim=0) - backward_work.mean(dim=0)
    lambdas = lambda_schedule_fn(
        torch.arange(n_rs_steps, device=batched_system.device),
        final_lambda=temperature_start / temperature_end,
        n_steps=n_rs_steps,
    )
    free_energy_rs = (
        free_energy_initial_temperature / lambdas[:-1]
        + 3
        / 2
        * temperature_start
        * ts.units.MetalUnits.temperature
        * torch.log(lambdas[:-1])
        / lambdas[:-1]
        + 1 / (2 * lambdas[:-1]) * (free_energy_difference)
    )

    return {
        # "forward_free_energy": forward_free_energy + free_energy_einstein,
        # "backward_free_energy": backward_free_energy + free_energy_einstein,
        "free_energy": free_energy_rs,
        "free_energy_difference": free_energy_difference,
        "lambdas": lambdas[:-1],
        "temperatures": temperature_start / lambdas[:-1],
        "forward_work": forward_work,
        "backward_work": backward_work,
        "forward_mean_work": forward_work.mean(),
        "forward_std_work": forward_work.std(),
        "backward_mean_work": backward_work.mean(),
        "backward_std_work": backward_work.std(),
    }
