"""Tests for MixedModel mixing functions and power_mixing helper."""

import torch

from torch_sim.models.lennard_jones import LennardJonesModel
from torch_sim.state import SimState, initialize_state
from torch_sim.workflows.steered_md import MixedModel, power_mixing

DEVICE = torch.device("cpu")
DTYPE = torch.float64


def _make_lj_model(epsilon: float = 0.0104) -> LennardJonesModel:
    return LennardJonesModel(
        sigma=3.405,
        epsilon=epsilon,
        device=DEVICE,
        dtype=DTYPE,
        compute_forces=True,
        compute_stress=False,
        cutoff=2.5 * 3.405,
    )


def _make_state() -> SimState:
    from ase.build import bulk

    atoms = bulk("Ar", "fcc", a=5.26, cubic=True)
    state = initialize_state(atoms, DEVICE, DTYPE)
    state.lambda_ = torch.tensor([0.3], dtype=DTYPE, device=DEVICE)
    return state


class TestMixedModelDefaultLinear:
    """Test that default MixedModel reproduces the old (1-λ)H₁ + λH₂ behavior."""

    def test_energy_is_linear_mix(self) -> None:
        model1 = _make_lj_model(epsilon=0.0104)
        model2 = _make_lj_model(epsilon=0.0208)
        state = _make_state()
        lam = state.lambda_[0]

        out1 = model1(state)
        out2 = model2(state)

        mixed = MixedModel(model1, model2, device=DEVICE, dtype=DTYPE)
        out = mixed(state)

        expected_energy = (1 - lam) * out1["energy"] + lam * out2["energy"]
        torch.testing.assert_close(out["energy"], expected_energy)

    def test_forces_are_linear_mix(self) -> None:
        model1 = _make_lj_model(epsilon=0.0104)
        model2 = _make_lj_model(epsilon=0.0208)
        state = _make_state()
        lam = state.lambda_[0]
        lam_per_atom = lam.expand(state.positions.shape[0]).view(-1, 1)

        out1 = model1(state)
        out2 = model2(state)

        mixed = MixedModel(model1, model2, device=DEVICE, dtype=DTYPE)
        out = mixed(state)

        expected_forces = (1 - lam_per_atom) * out1["forces"] + lam_per_atom * out2[
            "forces"
        ]
        torch.testing.assert_close(out["forces"], expected_forces)

    def test_energy_difference_is_h2_minus_h1(self) -> None:
        model1 = _make_lj_model(epsilon=0.0104)
        model2 = _make_lj_model(epsilon=0.0208)
        state = _make_state()

        out1 = model1(state)
        out2 = model2(state)

        mixed = MixedModel(model1, model2, device=DEVICE, dtype=DTYPE)
        out = mixed(state)

        expected_diff = out2["energy"] - out1["energy"]
        torch.testing.assert_close(out["energy_difference"], expected_diff)


class TestMixedModelPowerMixing:
    """Test power_mixing(m) produces correct energy and dH/dλ."""

    def test_power_mixing_m3_energy(self) -> None:
        model1 = _make_lj_model(epsilon=0.0104)
        model2 = _make_lj_model(epsilon=0.0208)
        state = _make_state()
        lam = state.lambda_[0]

        out1 = model1(state)
        out2 = model2(state)

        mixed = MixedModel(model1, model2, device=DEVICE, dtype=DTYPE, **power_mixing(3))
        out = mixed(state)

        expected_energy = (1 - lam) ** 3 * out1["energy"] + lam**3 * out2["energy"]
        torch.testing.assert_close(out["energy"], expected_energy)

    def test_power_mixing_m3_derivative(self) -> None:
        model1 = _make_lj_model(epsilon=0.0104)
        model2 = _make_lj_model(epsilon=0.0208)
        state = _make_state()
        lam = state.lambda_[0]

        out1 = model1(state)
        out2 = model2(state)

        mixed = MixedModel(model1, model2, device=DEVICE, dtype=DTYPE, **power_mixing(3))
        out = mixed(state)

        expected_diff = (
            -3 * (1 - lam) ** 2 * out1["energy"] + 3 * lam**2 * out2["energy"]
        )
        torch.testing.assert_close(out["energy_difference"], expected_diff)

    def test_power_mixing_m1_matches_default(self) -> None:
        """power_mixing(1) should match default linear mixing."""
        model1 = _make_lj_model(epsilon=0.0104)
        model2 = _make_lj_model(epsilon=0.0208)
        state = _make_state()

        default_mixed = MixedModel(model1, model2, device=DEVICE, dtype=DTYPE)
        power1_mixed = MixedModel(
            model1, model2, device=DEVICE, dtype=DTYPE, **power_mixing(1)
        )

        out_default = default_mixed(state)
        out_power1 = power1_mixed(state)

        torch.testing.assert_close(out_default["energy"], out_power1["energy"])
        torch.testing.assert_close(out_default["forces"], out_power1["forces"])
        torch.testing.assert_close(
            out_default["energy_difference"], out_power1["energy_difference"]
        )

    def test_power_mixing_forces(self) -> None:
        model1 = _make_lj_model(epsilon=0.0104)
        model2 = _make_lj_model(epsilon=0.0208)
        state = _make_state()
        lam = state.lambda_[0]
        lam_per_atom = lam.expand(state.positions.shape[0]).view(-1, 1)

        out1 = model1(state)
        out2 = model2(state)

        mixed = MixedModel(model1, model2, device=DEVICE, dtype=DTYPE, **power_mixing(3))
        out = mixed(state)

        expected_forces = (1 - lam_per_atom) ** 3 * out1[
            "forces"
        ] + lam_per_atom**3 * out2["forces"]
        torch.testing.assert_close(out["forces"], expected_forces)

    def test_power_mixing_at_endpoints(self) -> None:
        """At λ=0, H = H₁; at λ=1, H = H₂ regardless of exponent."""
        model1 = _make_lj_model(epsilon=0.0104)
        model2 = _make_lj_model(epsilon=0.0208)
        state = _make_state()

        out1 = model1(state)
        out2 = model2(state)

        mixed = MixedModel(model1, model2, device=DEVICE, dtype=DTYPE, **power_mixing(5))

        # λ=0 → pure model1
        state.lambda_ = torch.tensor([0.0], dtype=DTYPE, device=DEVICE)
        out = mixed(state)
        torch.testing.assert_close(out["energy"], out1["energy"])

        # λ=1 → pure model2
        state.lambda_ = torch.tensor([1.0], dtype=DTYPE, device=DEVICE)
        out = mixed(state)
        torch.testing.assert_close(out["energy"], out2["energy"])
