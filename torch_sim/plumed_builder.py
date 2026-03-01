"""Pythonic builder API for PLUMED collective variables and bias actions.

Provides CV classes and bias action classes that compose cleanly, handle
unit conversion automatically, and emit PLUMED input strings internally.
The raw-string interface in :mod:`torch_sim.plumed` remains fully intact.

Example::

    from torch_sim import Distance, Restraint, PlumedPrint, build_plumed_input

    d = Distance([0, 1])
    r = Restraint(d, center=3.726, kappa=100.0)  # Å and eV/Å²
    p = PlumedPrint([d], stride=10)

    lines = build_plumed_input([r, p])
    # ['d0: DISTANCE ATOMS=1,2',
    #  'RESTRAINT ARG=d0 AT=0.3726 KAPPA=964853.',
    #  'PRINT ARG=d0 STRIDE=10 FILE=COLVAR']

    plumed_model = ts.PlumedModel(model, plumed_input=[r, p], timestep=0.001)

Notes:
    All CV classes accept **TorchSim-native user units**:

    - Distances: Ångström (Å)
    - Angles and dihedrals *centre/sigma*: degrees
    - Angles and dihedrals *kappa*: eV/rad²
    - Distance *kappa*: eV/Å²
    - Energies (height): eV

    Atom indices are **0-based** in the Python API; the builder adds 1 when
    emitting PLUMED strings (which use 1-based indexing).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import Counter


# ---------------------------------------------------------------------------
# Unit conversion constants
# (same values as in torch_sim.plumed; defined here to avoid circular imports)
# ---------------------------------------------------------------------------

_PLUMED_LENGTH_FACTOR: float = 0.1  # 1 Å = 0.1 nm
_PLUMED_ENERGY_FACTOR: float = 96.4853321  # 1 eV = 96.4853321 kJ/mol


# ---------------------------------------------------------------------------
# Auto-naming counter
# ---------------------------------------------------------------------------

_CV_COUNTER: Counter[str] = Counter()


def _reset_cv_counter() -> None:
    """Reset the CV auto-naming counter.

    Intended for use in test fixtures to ensure test isolation.
    """
    _CV_COUNTER.clear()


# ---------------------------------------------------------------------------
# Collective Variable base class
# ---------------------------------------------------------------------------


class CollectiveVariable(ABC):
    """Abstract base class for PLUMED collective variables.

    Each subclass represents one PLUMED CV keyword (DISTANCE, ANGLE, …).
    Subclasses must implement :meth:`to_plumed_line`,
    :meth:`plumed_cv_unit_factor`, and :meth:`_auto_name_prefix`.

    Attributes:
        name: PLUMED label. Auto-generated from a prefix + counter if not
            supplied by the user.
    """

    def __init__(self, name: str | None = None) -> None:
        """Initialise a collective variable.

        Args:
            name: PLUMED label. Auto-generated (e.g. ``"d0"``, ``"a1"``) if
                ``None``.
        """
        if name is None:
            prefix = self._auto_name_prefix()
            idx = _CV_COUNTER[prefix]
            _CV_COUNTER[prefix] += 1
            self.name = f"{prefix}{idx}"
        else:
            self.name = name

    @abstractmethod
    def _auto_name_prefix(self) -> str:
        """Return the prefix used for auto-generated names (e.g. ``'d'``)."""

    @abstractmethod
    def to_plumed_line(self) -> str:
        """Return the PLUMED definition line for this CV.

        Example: ``"d0: DISTANCE ATOMS=1,2"``
        """

    @abstractmethod
    def plumed_cv_unit_factor(self) -> float:
        """Conversion factor from TorchSim user units to PLUMED units.

        Used for converting *centre* and *sigma* parameters. For example,
        :class:`Distance` returns ``0.1`` (Å → nm) and :class:`Angle`
        returns ``π/180`` (degrees → radians).

        Returns:
            float: Multiplicative factor applied to user-supplied CV values.
        """

    def plumed_kappa_sq_factor(self) -> float:
        """Squared position-unit factor used for kappa conversion.

        Returns the factor ``f²`` such that::

            kappa_plumed = kappa_user * _PLUMED_ENERGY_FACTOR / f²

        The default implementation returns
        ``plumed_cv_unit_factor() ** 2``, which is correct for CVs where
        the user specifies *kappa* in ``eV / (TorchSim_unit)²`` (e.g.
        eV/Å² for distances).

        For CVs where the user specifies *kappa* in ``eV/rad²`` (angles,
        dihedrals), override to return ``1.0``.

        Returns:
            float: Squared unit factor for kappa conversion.
        """
        return self.plumed_cv_unit_factor() ** 2


# ---------------------------------------------------------------------------
# CV implementations
# ---------------------------------------------------------------------------


class Distance(CollectiveVariable):
    """Distance between two atoms.

    The distance CV is defined in PLUMED as ``DISTANCE ATOMS=i,j``.

    User units:

    - Centre / sigma: Å
    - Kappa: eV/Å²

    Args:
        atoms: Two 0-based atom indices.
        name: PLUMED label. Auto-generated (e.g. ``"d0"``) if ``None``.

    Raises:
        ValueError: If ``atoms`` does not contain exactly 2 entries.

    Example::

        d = Distance([0, 1])
        print(d.to_plumed_line())  # "d0: DISTANCE ATOMS=1,2"
    """

    def __init__(self, atoms: list[int], name: str | None = None) -> None:
        """Initialise a Distance CV.

        Args:
            atoms: List of exactly two 0-based atom indices.
            name: PLUMED label. Auto-generated if ``None``.

        Raises:
            ValueError: If ``atoms`` does not have exactly 2 entries.
        """
        if len(atoms) != 2:
            raise ValueError(f"Distance requires exactly 2 atoms, got {len(atoms)}.")
        self._atoms = atoms
        super().__init__(name)

    def _auto_name_prefix(self) -> str:
        return "d"

    def to_plumed_line(self) -> str:
        """Return PLUMED DISTANCE definition line."""
        a1, a2 = self._atoms[0] + 1, self._atoms[1] + 1
        return f"{self.name}: DISTANCE ATOMS={a1},{a2}"

    def plumed_cv_unit_factor(self) -> float:
        """Return Å → nm factor (0.1)."""
        return _PLUMED_LENGTH_FACTOR


class Angle(CollectiveVariable):
    """Angle formed by three atoms.

    The angle CV is defined in PLUMED as ``ANGLE ATOMS=i,j,k``.

    User units:

    - Centre / sigma: degrees
    - Kappa: eV/rad²

    Args:
        atoms: Three 0-based atom indices.
        name: PLUMED label. Auto-generated (e.g. ``"a0"``) if ``None``.

    Raises:
        ValueError: If ``atoms`` does not contain exactly 3 entries.
    """

    def __init__(self, atoms: list[int], name: str | None = None) -> None:
        """Initialise an Angle CV.

        Args:
            atoms: List of exactly three 0-based atom indices.
            name: PLUMED label. Auto-generated if ``None``.

        Raises:
            ValueError: If ``atoms`` does not have exactly 3 entries.
        """
        if len(atoms) != 3:
            raise ValueError(f"Angle requires exactly 3 atoms, got {len(atoms)}.")
        self._atoms = atoms
        super().__init__(name)

    def _auto_name_prefix(self) -> str:
        return "a"

    def to_plumed_line(self) -> str:
        """Return PLUMED ANGLE definition line."""
        a1, a2, a3 = (a + 1 for a in self._atoms)
        return f"{self.name}: ANGLE ATOMS={a1},{a2},{a3}"

    def plumed_cv_unit_factor(self) -> float:
        """Return degrees → radians factor (π/180)."""
        return math.pi / 180.0

    def plumed_kappa_sq_factor(self) -> float:
        """Return 1.0 — kappa in eV/rad² → kJ/mol/rad² needs only energy factor."""
        return 1.0


class Dihedral(CollectiveVariable):
    """Dihedral (torsion) angle formed by four atoms.

    The dihedral CV is defined in PLUMED as ``TORSION ATOMS=i,j,k,l``.

    User units:

    - Centre / sigma: degrees
    - Kappa: eV/rad²

    Args:
        atoms: Four 0-based atom indices.
        name: PLUMED label. Auto-generated (e.g. ``"phi0"``) if ``None``.

    Raises:
        ValueError: If ``atoms`` does not contain exactly 4 entries.
    """

    def __init__(self, atoms: list[int], name: str | None = None) -> None:
        """Initialise a Dihedral CV.

        Args:
            atoms: List of exactly four 0-based atom indices.
            name: PLUMED label. Auto-generated if ``None``.

        Raises:
            ValueError: If ``atoms`` does not have exactly 4 entries.
        """
        if len(atoms) != 4:
            raise ValueError(f"Dihedral requires exactly 4 atoms, got {len(atoms)}.")
        self._atoms = atoms
        super().__init__(name)

    def _auto_name_prefix(self) -> str:
        return "phi"

    def to_plumed_line(self) -> str:
        """Return PLUMED TORSION definition line."""
        atom_str = ",".join(str(a + 1) for a in self._atoms)
        return f"{self.name}: TORSION ATOMS={atom_str}"

    def plumed_cv_unit_factor(self) -> float:
        """Return degrees → radians factor (π/180)."""
        return math.pi / 180.0

    def plumed_kappa_sq_factor(self) -> float:
        """Return 1.0 — kappa in eV/rad² → kJ/mol/rad² needs only energy factor."""
        return 1.0


class Coordination(CollectiveVariable):
    """Coordination number between two groups of atoms.

    Uses PLUMED's rational switching function
    ``s(r) = (1 - (r/r0)^nn) / (1 - (r/r0)^mm)``.

    User units:

    - ``r0``, ``d0``: Å

    Args:
        group1: 0-based indices for the first atom group.
        group2: 0-based indices for the second atom group.
        r0: Reference distance in Å.
        nn: Numerator exponent. Defaults to 6.
        mm: Denominator exponent (0 = PLUMED auto-selects 2xnn). Defaults to 0.
        d0: Offset distance in Å. Defaults to 0.0.
        name: PLUMED label. Auto-generated (e.g. ``"c0"``) if ``None``.
    """

    def __init__(
        self,
        group1: list[int],
        group2: list[int],
        r0: float,
        nn: int = 6,
        mm: int = 0,
        d0: float = 0.0,
        name: str | None = None,
    ) -> None:
        """Initialise a Coordination CV.

        Args:
            group1: 0-based atom indices for the first group.
            group2: 0-based atom indices for the second group.
            r0: Reference distance in Å.
            nn: Numerator exponent in switching function. Defaults to 6.
            mm: Denominator exponent in switching function (0 = 2xnn).
                Defaults to 0.
            d0: Offset distance in Å. Defaults to 0.0.
            name: PLUMED label. Auto-generated if ``None``.
        """
        self._group1 = group1
        self._group2 = group2
        self._r0 = r0
        self._nn = nn
        self._mm = mm
        self._d0 = d0
        super().__init__(name)

    def _auto_name_prefix(self) -> str:
        return "c"

    def to_plumed_line(self) -> str:
        """Return PLUMED COORDINATION definition line."""
        g1 = ",".join(str(a + 1) for a in self._group1)
        g2 = ",".join(str(a + 1) for a in self._group2)
        r0_nm = self._r0 * _PLUMED_LENGTH_FACTOR
        line = (
            f"{self.name}: COORDINATION GROUPA={g1} GROUPB={g2}"
            f" R_0={r0_nm:.6g} NN={self._nn}"
        )
        if self._mm != 0:
            line += f" MM={self._mm}"
        if self._d0 != 0.0:
            d0_nm = self._d0 * _PLUMED_LENGTH_FACTOR
            line += f" D_0={d0_nm:.6g}"
        return line

    def plumed_cv_unit_factor(self) -> float:
        """Return Å → nm factor (0.1)."""
        return _PLUMED_LENGTH_FACTOR


class Gyration(CollectiveVariable):
    """Radius of gyration of a group of atoms.

    The gyration CV is defined in PLUMED as
    ``GYRATION TYPE=<type> ATOMS=i,j,...``.

    PLUMED returns this CV in nm. Users specifying centre/sigma for walls or
    restraints should provide values in nm directly (unit factor = 1.0).

    Args:
        atoms: 0-based atom indices.
        type: Gyration type. One of ``"RADIUS"``, ``"TRACE"``,
            ``"GTENSOR"``, ``"ASPHERICITY"``, ``"ACYLINDRICITY"``, or
            ``"KAPPA2"``. Defaults to ``"RADIUS"``.
        name: PLUMED label. Auto-generated (e.g. ``"g0"``) if ``None``.
    """

    def __init__(
        self,
        atoms: list[int],
        type: str = "RADIUS",  # noqa: A002
        name: str | None = None,
    ) -> None:
        """Initialise a Gyration CV.

        Args:
            atoms: 0-based atom indices.
            type: Gyration type (``"RADIUS"``, ``"TRACE"``, ``"GTENSOR"``,
                ``"ASPHERICITY"``, ``"ACYLINDRICITY"``, ``"KAPPA2"``).
                Defaults to ``"RADIUS"``.
            name: PLUMED label. Auto-generated if ``None``.
        """
        self._atoms = atoms
        self._type = type
        super().__init__(name)

    def _auto_name_prefix(self) -> str:
        return "g"

    def to_plumed_line(self) -> str:
        """Return PLUMED GYRATION definition line."""
        atom_str = ",".join(str(a + 1) for a in self._atoms)
        return f"{self.name}: GYRATION TYPE={self._type} ATOMS={atom_str}"

    def plumed_cv_unit_factor(self) -> float:
        """Return 1.0 — gyration is output in nm; use nm for centre/sigma."""
        return 1.0


# ---------------------------------------------------------------------------
# Bias Action base class
# ---------------------------------------------------------------------------


class BiasAction(ABC):
    """Abstract base class for PLUMED bias actions.

    Bias actions (RESTRAINT, METAD, UPPER_WALLS, …) reference one or more
    :class:`CollectiveVariable` objects and emit PLUMED command strings.
    """

    @abstractmethod
    def collect_cvs(self) -> list[CollectiveVariable]:
        """Return all CVs referenced by this action, in order of appearance.

        Used by :func:`build_plumed_input` to emit CV definition lines once,
        before the action lines.

        Returns:
            list[CollectiveVariable]: Ordered list of referenced CVs.
        """

    @abstractmethod
    def to_plumed_lines(self) -> list[str]:
        """Return PLUMED command strings for this action.

        Returns:
            list[str]: One or more PLUMED input lines (no newlines).
        """


# ---------------------------------------------------------------------------
# Action implementations
# ---------------------------------------------------------------------------


class Restraint(BiasAction):
    """Harmonic restraint on a collective variable.

    Adds the bias ``V(xi) = k/2 * (xi - xi0)**2`` to the potential energy.

    Args:
        cv: Collective variable to restrain.
        center: Restraint centre in CV user units (Å for Distance,
            degrees for Angle/Dihedral).
        kappa: Force constant. eV/Å² for Distance; eV/rad² for
            Angle/Dihedral.

    Example::

        d = Distance([0, 1])
        r = Restraint(d, center=3.5, kappa=100.0)  # 3.5 Å, 100 eV/Å²
        print(r.to_plumed_lines())
        # ['RESTRAINT ARG=d0 AT=0.35 KAPPA=964853.']
    """

    def __init__(self, cv: CollectiveVariable, center: float, kappa: float) -> None:
        """Initialise a Restraint.

        Args:
            cv: Collective variable to restrain.
            center: Restraint centre in CV user units.
            kappa: Force constant in eV/Å² (Distance) or eV/rad²
                (Angle/Dihedral).
        """
        self._cv = cv
        self._center = center
        self._kappa = kappa

    def collect_cvs(self) -> list[CollectiveVariable]:
        """Return the CV referenced by this restraint."""
        return [self._cv]

    def to_plumed_lines(self) -> list[str]:
        """Return PLUMED RESTRAINT line with converted units."""
        center_plumed = self._center * self._cv.plumed_cv_unit_factor()
        kappa_plumed = (
            self._kappa * _PLUMED_ENERGY_FACTOR / self._cv.plumed_kappa_sq_factor()
        )
        return [
            f"RESTRAINT ARG={self._cv.name}"
            f" AT={center_plumed:.6g} KAPPA={kappa_plumed:.6g}"
        ]


class UpperWall(BiasAction):
    """Upper wall: harmonic penalty applied only when CV > at.

    Emits ``UPPER_WALLS ARG=<label> AT=<at> KAPPA=<kappa> EXP=<exp>``.

    Args:
        cv: Collective variable.
        at: Wall position in CV user units (Å for Distance, degrees for
            Angle/Dihedral).
        kappa: Wall force constant. eV/Å² for Distance; eV/rad² for
            Angle/Dihedral.
        exp: Wall exponent. Defaults to 2.
    """

    def __init__(
        self, cv: CollectiveVariable, at: float, kappa: float, exp: int = 2
    ) -> None:
        """Initialise an UpperWall.

        Args:
            cv: Collective variable.
            at: Wall position in CV user units.
            kappa: Force constant in eV/Å² (Distance) or eV/rad²
                (Angle/Dihedral).
            exp: Wall exponent. Defaults to 2.
        """
        self._cv = cv
        self._at = at
        self._kappa = kappa
        self._exp = exp

    def collect_cvs(self) -> list[CollectiveVariable]:
        """Return the CV referenced by this wall."""
        return [self._cv]

    def to_plumed_lines(self) -> list[str]:
        """Return PLUMED UPPER_WALLS line with converted units."""
        at_plumed = self._at * self._cv.plumed_cv_unit_factor()
        kappa_plumed = (
            self._kappa * _PLUMED_ENERGY_FACTOR / self._cv.plumed_kappa_sq_factor()
        )
        return [
            f"UPPER_WALLS ARG={self._cv.name}"
            f" AT={at_plumed:.6g} KAPPA={kappa_plumed:.6g} EXP={self._exp}"
        ]


class LowerWall(BiasAction):
    """Lower wall: harmonic penalty applied only when CV < at.

    Emits ``LOWER_WALLS ARG=<label> AT=<at> KAPPA=<kappa> EXP=<exp>``.

    Args:
        cv: Collective variable.
        at: Wall position in CV user units (Å for Distance, degrees for
            Angle/Dihedral).
        kappa: Wall force constant. eV/Å² for Distance; eV/rad² for
            Angle/Dihedral.
        exp: Wall exponent. Defaults to 2.
    """

    def __init__(
        self, cv: CollectiveVariable, at: float, kappa: float, exp: int = 2
    ) -> None:
        """Initialise a LowerWall.

        Args:
            cv: Collective variable.
            at: Wall position in CV user units.
            kappa: Force constant in eV/Å² (Distance) or eV/rad²
                (Angle/Dihedral).
            exp: Wall exponent. Defaults to 2.
        """
        self._cv = cv
        self._at = at
        self._kappa = kappa
        self._exp = exp

    def collect_cvs(self) -> list[CollectiveVariable]:
        """Return the CV referenced by this wall."""
        return [self._cv]

    def to_plumed_lines(self) -> list[str]:
        """Return PLUMED LOWER_WALLS line with converted units."""
        at_plumed = self._at * self._cv.plumed_cv_unit_factor()
        kappa_plumed = (
            self._kappa * _PLUMED_ENERGY_FACTOR / self._cv.plumed_kappa_sq_factor()
        )
        return [
            f"LOWER_WALLS ARG={self._cv.name}"
            f" AT={at_plumed:.6g} KAPPA={kappa_plumed:.6g} EXP={self._exp}"
        ]


class Metadynamics(BiasAction):
    """Well-tempered or standard metadynamics.

    Deposits Gaussian hills along one or more CVs. For well-tempered
    metadynamics pass a ``biasfactor > 1``.

    Args:
        cvs: CV or list of CVs to bias (1D or 2D metadynamics).
        sigma: Gaussian width in CV user units (Å for Distance, degrees for
            Angle). Pass a list for multi-CV metadynamics.
        height: Gaussian height in eV.
        pace: Number of MD steps between Gaussian depositions.
        biasfactor: Well-tempering bias factor (must be > 1). ``None`` for
            standard (non-tempered) metadynamics. Defaults to ``None``.
        file: HILLS output file name. Defaults to ``"HILLS"``.

    Example::

        d = Distance([0, 1])
        metad = Metadynamics(d, sigma=0.5, height=0.005, pace=100, biasfactor=10.0)
    """

    def __init__(
        self,
        cvs: CollectiveVariable | list[CollectiveVariable],
        sigma: float | list[float],
        height: float,
        pace: int,
        biasfactor: float | None = None,
        file: str = "HILLS",
    ) -> None:
        """Initialise Metadynamics.

        Args:
            cvs: CV or list of CVs to bias.
            sigma: Gaussian width in CV user units. Single float for 1D;
                list for multi-CV.
            height: Gaussian height in eV.
            pace: Steps between hill depositions.
            biasfactor: Well-tempering bias factor (> 1). ``None`` for
                standard metadynamics. Defaults to ``None``.
            file: HILLS output file name. Defaults to ``"HILLS"``.
        """
        self._cvs: list[CollectiveVariable] = (
            [cvs] if isinstance(cvs, CollectiveVariable) else list(cvs)
        )
        self._sigma: list[float] = (
            [sigma] if isinstance(sigma, (int, float)) else list(sigma)
        )
        self._height = height
        self._pace = pace
        self._biasfactor = biasfactor
        self._file = file

    def collect_cvs(self) -> list[CollectiveVariable]:
        """Return all CVs referenced by this metadynamics action."""
        return list(self._cvs)

    def to_plumed_lines(self) -> list[str]:
        """Return PLUMED METAD line with converted units."""
        arg = ",".join(cv.name for cv in self._cvs)
        sigma_plumed = ",".join(
            f"{s * cv.plumed_cv_unit_factor():.6g}"
            for s, cv in zip(self._sigma, self._cvs, strict=False)
        )
        height_plumed = self._height * _PLUMED_ENERGY_FACTOR
        line = (
            f"METAD ARG={arg} SIGMA={sigma_plumed}"
            f" HEIGHT={height_plumed:.6g} PACE={self._pace}"
        )
        if self._biasfactor is not None:
            line += f" BIASFACTOR={self._biasfactor:.6g}"
        line += f" FILE={self._file}"
        return [line]


class Print(BiasAction):
    """PLUMED PRINT action: write CV values to a file.

    Accepts :class:`CollectiveVariable` objects or raw PLUMED label strings
    (e.g. ``"metad.bias"`` to include a METAD bias value in the output).

    Args:
        cvs: CVs or raw PLUMED label strings to print.
        stride: Output frequency in MD steps. Defaults to 1.
        file: Output file name. Defaults to ``"COLVAR"``. When used inside
            :class:`~torch_sim.plumed.PlumedModel` with shared input and
            multiple systems, this name is automatically suffixed with
            ``.{i}`` to keep each system's output separate.

    Example::

        d = Distance([0, 1])
        p = Print([d, "metad.bias"], stride=10, file="COLVAR")
    """

    def __init__(
        self,
        cvs: list[CollectiveVariable | str],
        stride: int = 1,
        file: str = "COLVAR",
    ) -> None:
        """Initialise a Print action.

        Args:
            cvs: List of CVs or raw PLUMED label strings (e.g.
                ``"metad.bias"``).
            stride: Write every ``stride`` steps. Defaults to 1.
            file: Output file name. Defaults to ``"COLVAR"``.
        """
        self._cvs = cvs
        self._stride = stride
        self._file = file

    def collect_cvs(self) -> list[CollectiveVariable]:
        """Return only CollectiveVariable objects (string labels are skipped)."""
        return [cv for cv in self._cvs if isinstance(cv, CollectiveVariable)]

    def to_plumed_lines(self) -> list[str]:
        """Return PLUMED PRINT line."""
        labels = [
            cv.name if isinstance(cv, CollectiveVariable) else cv for cv in self._cvs
        ]
        arg = ",".join(labels)
        return [f"PRINT ARG={arg} STRIDE={self._stride} FILE={self._file}"]


# ---------------------------------------------------------------------------
# Top-level builder helper
# ---------------------------------------------------------------------------


def build_plumed_input(actions: list[BiasAction]) -> list[str]:
    """Build a flat PLUMED input string list from a list of bias actions.

    CV definition lines are emitted exactly once, in the order they are first
    encountered while walking through the actions' :meth:`~BiasAction.collect_cvs`
    calls. Action lines follow immediately after the CV definitions they depend on.

    Args:
        actions: Ordered list of :class:`BiasAction` objects.

    Returns:
        list[str]: Ready-to-use PLUMED command strings, suitable for passing
        directly to :class:`~torch_sim.plumed.PlumedModel`.

    Example::

        d = Distance([0, 1])
        r = Restraint(d, center=3.5, kappa=100.0)
        p = Print([d], stride=10)
        lines = build_plumed_input([r, p])
        # ['d0: DISTANCE ATOMS=1,2',
        #  'RESTRAINT ARG=d0 AT=0.35 KAPPA=964853.',
        #  'PRINT ARG=d0 STRIDE=10 FILE=COLVAR']
    """
    lines: list[str] = []
    seen: set[str] = set()
    for action in actions:
        for cv in action.collect_cvs():
            if cv.name not in seen:
                lines.append(cv.to_plumed_line())
                seen.add(cv.name)
        lines.extend(action.to_plumed_lines())
    return lines
