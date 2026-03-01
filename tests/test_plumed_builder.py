"""Tests for torch_sim/plumed_builder.py.

These tests exercise the Pythonic PLUMED builder API (CV classes, action
classes, and build_plumed_input).  No PLUMED kernel is required — all tests
are pure Python string-generation tests.
"""

from __future__ import annotations

import math
from unittest import mock

import pytest

from torch_sim.plumed_builder import (
    _PLUMED_ENERGY_FACTOR,
    _PLUMED_LENGTH_FACTOR,
    Angle,
    Coordination,
    Dihedral,
    Distance,
    Gyration,
    LowerWall,
    Metadynamics,
    Print,
    Restraint,
    UpperWall,
    _reset_cv_counter,
    build_plumed_input,
)


@pytest.fixture(autouse=True)
def reset_counter() -> None:
    """Reset the CV auto-naming counter before each test for isolation."""
    _reset_cv_counter()


# ---------------------------------------------------------------------------
# CV class tests
# ---------------------------------------------------------------------------


def test_distance_to_plumed_line() -> None:
    """Distance CV emits DISTANCE keyword with 1-based atom indices."""
    d = Distance([0, 1])
    line = d.to_plumed_line()
    assert line.startswith("d0:")
    assert "DISTANCE" in line
    assert "ATOMS=1,2" in line


def test_angle_degree_to_radian() -> None:
    """Angle CV converts centre from degrees to radians in PLUMED output."""
    a = Angle([0, 1, 2])
    assert a.plumed_cv_unit_factor() == pytest.approx(math.pi / 180.0)

    r = Restraint(a, center=90.0, kappa=10.0)
    lines = r.to_plumed_lines()
    expected_at = 90.0 * math.pi / 180.0  # π/2 ≈ 1.5708
    # Check the AT= value appears in the line
    assert f"AT={expected_at:.6g}" in lines[0]


def test_dihedral_to_plumed_line() -> None:
    """Dihedral CV emits TORSION keyword with 1-based atom indices."""
    phi = Dihedral([0, 1, 2, 3])
    line = phi.to_plumed_line()
    assert "TORSION" in line
    assert "ATOMS=1,2,3,4" in line
    assert line.startswith("phi0:")


def test_auto_naming() -> None:
    """Second Distance() in the same session gets an incremented suffix."""
    d0 = Distance([0, 1])
    d1 = Distance([1, 2])
    assert d0.name == "d0"
    assert d1.name == "d1"
    assert d0.name != d1.name


def test_custom_name() -> None:
    """User-supplied name overrides auto-generated label."""
    d = Distance([0, 1], name="myCV")
    assert d.name == "myCV"
    assert d.to_plumed_line().startswith("myCV:")
    assert "DISTANCE" in d.to_plumed_line()


def test_coordination_to_plumed_line() -> None:
    """Coordination CV emits COORDINATION keyword with correct group labels."""
    c = Coordination([0, 1], [2, 3], r0=3.5)
    line = c.to_plumed_line()
    assert "COORDINATION" in line
    assert "GROUPA=1,2" in line
    assert "GROUPB=3,4" in line
    # r0 converted from Å to nm: 3.5 * 0.1 = 0.35
    assert "R_0=0.35" in line


def test_gyration_to_plumed_line() -> None:
    """Gyration CV emits GYRATION keyword with the requested TYPE."""
    g = Gyration([0, 1, 2, 3], type="RADIUS")
    line = g.to_plumed_line()
    assert "GYRATION" in line
    assert "TYPE=RADIUS" in line
    assert "ATOMS=1,2,3,4" in line


# ---------------------------------------------------------------------------
# Unit conversion tests
# ---------------------------------------------------------------------------


def test_restraint_unit_conversion() -> None:
    """Restraint converts distance centre (Å→nm) and kappa (eV/Å²→kJ/mol/nm²)."""
    d = Distance([0, 1])
    center_ang = 3.0  # Å
    kappa_ev = 100.0  # eV/Å²
    r = Restraint(d, center=center_ang, kappa=kappa_ev)
    lines = r.to_plumed_lines()
    assert len(lines) == 1

    # centre: 3.0 Å * 0.1 = 0.3 nm
    expected_at = center_ang * _PLUMED_LENGTH_FACTOR
    assert f"AT={expected_at:.6g}" in lines[0]

    # kappa: 100 eV/Å² * 96.4853321 kJ·mol⁻¹/eV / (0.1)² nm²/Å² = 964853.321
    expected_kappa = kappa_ev * _PLUMED_ENERGY_FACTOR / _PLUMED_LENGTH_FACTOR**2
    assert f"KAPPA={expected_kappa:.6g}" in lines[0]


def test_angle_kappa_unit_conversion() -> None:
    """Angle kappa (eV/rad²) → kJ/mol/rad² uses only the energy factor."""
    a = Angle([0, 1, 2])
    kappa_ev_rad2 = 50.0  # eV/rad²
    r = Restraint(a, center=90.0, kappa=kappa_ev_rad2)
    lines = r.to_plumed_lines()
    expected_kappa = kappa_ev_rad2 * _PLUMED_ENERGY_FACTOR  # /1.0 (rad is dimensionless)
    assert f"KAPPA={expected_kappa:.6g}" in lines[0]


def test_metadynamics_output() -> None:
    """Metadynamics emits correct METAD line with well-tempering parameters."""
    d = Distance([0, 1])
    metad = Metadynamics(d, sigma=0.5, height=0.01, pace=100, biasfactor=10.0)
    lines = metad.to_plumed_lines()
    assert len(lines) == 1
    assert "METAD" in lines[0]
    assert "BIASFACTOR=10" in lines[0]
    assert "PACE=100" in lines[0]
    assert "FILE=HILLS" in lines[0]
    # sigma: 0.5 Å * 0.1 = 0.05 nm
    expected_sigma = 0.5 * _PLUMED_LENGTH_FACTOR
    assert f"SIGMA={expected_sigma:.6g}" in lines[0]
    # height: 0.01 eV * 96.4853321 = 0.964853321 kJ/mol
    expected_height = 0.01 * _PLUMED_ENERGY_FACTOR
    assert f"HEIGHT={expected_height:.6g}" in lines[0]


def test_upper_wall_unit_conversion() -> None:
    """UpperWall converts position and kappa to PLUMED units."""
    d = Distance([0, 1])
    wall = UpperWall(d, at=5.0, kappa=200.0, exp=4)
    lines = wall.to_plumed_lines()
    assert "UPPER_WALLS" in lines[0]
    assert "EXP=4" in lines[0]
    expected_at = 5.0 * _PLUMED_LENGTH_FACTOR
    assert f"AT={expected_at:.6g}" in lines[0]


def test_lower_wall_unit_conversion() -> None:
    """LowerWall converts position and kappa to PLUMED units."""
    d = Distance([0, 1])
    wall = LowerWall(d, at=2.0, kappa=200.0)
    lines = wall.to_plumed_lines()
    assert "LOWER_WALLS" in lines[0]
    expected_at = 2.0 * _PLUMED_LENGTH_FACTOR
    assert f"AT={expected_at:.6g}" in lines[0]


# ---------------------------------------------------------------------------
# build_plumed_input ordering tests
# ---------------------------------------------------------------------------


def test_build_plumed_input_ordering() -> None:
    """CV definition lines appear before action lines."""
    d = Distance([0, 1])
    r = Restraint(d, center=3.5, kappa=100.0)
    p = Print([d], stride=10)
    lines = build_plumed_input([r, p])

    assert len(lines) == 3
    assert lines[0].startswith("d0:")  # CV first
    assert "DISTANCE" in lines[0]
    assert "RESTRAINT" in lines[1]
    assert "PRINT" in lines[2]


def test_build_plumed_input_shared_cv() -> None:
    """A CV shared between Restraint and Print appears only once."""
    d = Distance([0, 1])
    r = Restraint(d, center=3.5, kappa=100.0)
    p = Print([d], stride=10)
    lines = build_plumed_input([r, p])

    cv_definition_lines = [ln for ln in lines if ln.startswith("d0:")]
    assert len(cv_definition_lines) == 1


def test_build_plumed_input_multiple_cvs() -> None:
    """Multiple distinct CVs each get exactly one definition line."""
    d = Distance([0, 1])
    a = Angle([0, 1, 2])
    r1 = Restraint(d, center=3.5, kappa=100.0)
    r2 = Restraint(a, center=90.0, kappa=10.0)
    p = Print([d, a], stride=5)
    lines = build_plumed_input([r1, r2, p])

    dist_lines = [ln for ln in lines if "DISTANCE" in ln]
    angle_lines = [ln for ln in lines if "ANGLE" in ln]
    assert len(dist_lines) == 1
    assert len(angle_lines) == 1


def test_build_plumed_input_string_label_in_print() -> None:
    """Raw string labels (e.g. 'metad.bias') pass through Print unchanged."""
    d = Distance([0, 1])
    metad = Metadynamics(d, sigma=0.5, height=0.01, pace=100)
    p = Print([d, "metad.bias"], stride=1)
    lines = build_plumed_input([metad, p])

    print_line = next(ln for ln in lines if "PRINT" in ln)
    assert "metad.bias" in print_line


# ---------------------------------------------------------------------------
# Integration-style test: PlumedModel normalization (no kernel required)
# ---------------------------------------------------------------------------


def test_plumed_model_accepts_actions() -> None:
    """PlumedModel normalizes a list of BiasActions to strings at init time."""
    from torch_sim.plumed import PlumedModel

    d = Distance([0, 1])
    actions = [
        Restraint(d, center=3.5, kappa=100.0),
        Print([d], stride=10),
    ]

    # We need a minimal model to satisfy PlumedModel. Use a mock.
    model = mock.MagicMock()
    pm = PlumedModel(model=model, plumed_input=actions, timestep=0.001, kT=0.0257)

    # After normalization, plumed_input should be a plain list[str]
    assert isinstance(pm.plumed_input, list)
    assert all(isinstance(line, str) for line in pm.plumed_input)
    assert any("DISTANCE" in line for line in pm.plumed_input)
    assert any("RESTRAINT" in line for line in pm.plumed_input)


def test_multi_window_per_system_actions() -> None:
    """list-of-lists of BiasActions generates correct per-system string lists."""
    centers = [3.0, 3.5, 4.0]  # Å
    windows = [
        [
            Restraint(Distance([0, 1], name=f"d_w{i}"), center=c, kappa=100.0),
            Print([Distance([0, 1], name=f"d_w{i}")], file=f"COLVAR_w{i}"),
        ]
        for i, c in enumerate(centers)
    ]

    # Use build_plumed_input per window (mirrors what PlumedModel does internally)
    per_system_lines = [build_plumed_input(w) for w in windows]

    assert len(per_system_lines) == len(centers)
    for i, lines in enumerate(per_system_lines):
        assert any("DISTANCE" in ln for ln in lines)
        assert any("RESTRAINT" in ln for ln in lines)
        assert any(f"COLVAR_w{i}" in ln for ln in lines)
