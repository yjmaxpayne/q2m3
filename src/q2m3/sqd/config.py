"""Dependency-light SQD configurations and shared structural input validation."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field, fields, is_dataclass
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any, Literal

import numpy as np

from q2m3.molecule import MoleculeConfig
from q2m3.sqd.exceptions import CCSDConvergenceError, ProvenanceMismatchError

Tier = Literal["T0", "T1", "T1+", "T2"]
RunMode = Literal["full", "reference_only"]
JsonSnapshot = Mapping[str, Any]


def _freeze_snapshot(value: Any) -> Any:
    """Copy a finite JSON-compatible value into recursively immutable containers."""
    if is_dataclass(value) and not isinstance(value, type):
        value = {item.name: getattr(value, item.name) for item in fields(value)}
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("Snapshot mapping keys must be strings")
        return MappingProxyType({key: _freeze_snapshot(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze_snapshot(item) for item in value)
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float) and np.isfinite(value):
        return value
    raise ValueError("Snapshots require finite JSON-compatible values")


def _integer(value: Any, name: str, minimum: int = 0) -> None:
    if isinstance(value, bool | np.bool_) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _finite(value: Any, name: str) -> None:
    if isinstance(value, bool | np.bool_) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")


def _real_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in "iuf" or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite real array of shape {shape}")
    return array


def validate_active_space(norb: int, nelec: tuple[int, int]) -> None:
    """Validate the closed-shell orbital and spin-population domain.

    Args:
        norb: Number of active spatial orbitals.
        nelec: Alpha and beta electron populations.

    Raises:
        ValueError: Counts are nonintegral, unbalanced, or outside orbital capacity.
    """
    _integer(norb, "norb", 1)
    if not isinstance(nelec, tuple | list) or len(nelec) != 2:
        raise ValueError("nelec must contain two spin populations")
    for count in nelec:
        _integer(count, "nelec", 1)
    if nelec[0] != nelec[1] or nelec[0] > norb:
        raise ValueError("SQD requires a closed-shell active space within orbital capacity")


@dataclass(frozen=True)
class LUCJConfig:
    """Spin-balanced ansatz and sampling configuration."""

    n_reps: int = 2
    shots: int = 100_000
    interaction_pairs: tuple[tuple[tuple[int, int], ...] | None, ...] | None = None

    def __post_init__(self) -> None:
        if self.interaction_pairs is not None:
            try:
                pairs = tuple(
                    None if channel is None else tuple(tuple(p) for p in channel)
                    for channel in self.interaction_pairs
                )
            except TypeError as exc:
                raise ValueError("Invalid interaction_pairs structure") from exc
            object.__setattr__(self, "interaction_pairs", pairs)

    def validate(self) -> LUCJConfig:
        """Validate counts and connectivity structure, returning this instance."""
        _integer(self.n_reps, "n_reps", 1)
        _integer(self.shots, "shots", 1)
        if self.interaction_pairs is not None:
            if len(self.interaction_pairs) != 2:
                raise ValueError("interaction_pairs requires two spin-balanced channels")
            for channel in self.interaction_pairs:
                if channel is None:
                    continue
                seen = set()
                for pair in channel:
                    if len(pair) != 2:
                        raise ValueError("Each interaction pair must have two indices")
                    for index in pair:
                        _integer(index, "interaction index")
                    if pair[0] > pair[1] or pair in seen:
                        raise ValueError("Lower triangular or duplicate interaction pair")
                    seen.add(pair)
        return self


def _validate_pairs(lucj: LUCJConfig, norb: int) -> None:
    lucj.validate()
    if lucj.interaction_pairs is not None:
        for channel in lucj.interaction_pairs:
            if channel is not None and any(j >= norb for _, j in channel):
                raise ValueError("Interaction index exceeds the active orbital space")


@dataclass(frozen=True)
class ReferenceConfig:
    """Reference solver budgets, allowed tiers, and ordered optional plugins."""

    wall_budget_s: float = 900.0
    rss_budget_mb: float = 8192.0
    allowed_tiers: tuple[Tier, ...] = ("T0", "T1", "T1+", "T2")
    plugins: tuple[str, ...] = ()
    sci_cutoffs: tuple[float, float] = (1e-4, 1e-5)

    def __post_init__(self) -> None:
        for name in ("allowed_tiers", "plugins", "sci_cutoffs"):
            try:
                object.__setattr__(self, name, tuple(getattr(self, name)))
            except TypeError as exc:
                raise ValueError(f"{name} must be a sequence") from exc

    def validate(self) -> ReferenceConfig:
        """Validate finite budgets and strictly descending cutoffs; return self."""
        for name in ("wall_budget_s", "rss_budget_mb"):
            _finite(getattr(self, name), name)
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name, supported in (
            ("allowed_tiers", ("T0", "T1", "T1+", "T2")),
            ("plugins", ("shci_dice", "dmrg_block2")),
        ):
            values = getattr(self, name)
            if any(value not in supported for value in values) or len(set(values)) != len(values):
                raise ValueError(f"Invalid or duplicate {name}")
        if "T0" not in self.allowed_tiers:
            raise ValueError("allowed_tiers must include T0")
        if len(self.sci_cutoffs) != 2:
            raise ValueError("sci_cutoffs requires two cutoffs")
        for cutoff in self.sci_cutoffs:
            _finite(cutoff, "cutoff")
        if not 0 < self.sci_cutoffs[1] < self.sci_cutoffs[0] < 1:
            raise ValueError("sci_cutoffs must be strictly descending within (0, 1)")
        return self


def validate_molecular_inputs(
    molecule: MoleculeConfig,
    *,
    mm_charges: np.ndarray | None = None,
    mm_coords: np.ndarray | None = None,
    embedding_mode: str = "diagonal",
) -> None:
    """Validate the shared molecular and optional MM input domain.

    Args:
        molecule: Molecular geometry in Angstrom and explicit active-space counts.
        mm_charges: Optional point charges in elementary charge units.
        mm_coords: Matching point-charge coordinates in Angstrom.
        embedding_mode: Diagonal or full one-electron MM correction.

    Raises:
        ValueError: Geometry, active-space, or MM data violate the supported domain.
    """
    if not isinstance(molecule, MoleculeConfig):
        raise ValueError("molecule must be a MoleculeConfig")
    molecule.validate()
    if not molecule.symbols or any(
        not isinstance(s, str) or not s.strip() for s in molecule.symbols
    ):
        raise ValueError("Atomic symbols must be nonempty strings")
    _real_array(molecule.coords, (len(molecule.symbols), 3), "coordinates")
    if isinstance(molecule.charge, bool) or not isinstance(molecule.charge, Integral):
        raise ValueError("charge must be an integer")
    _integer(molecule.active_electrons, "active_electrons", 1)
    if molecule.active_electrons % 2:
        raise ValueError("SQD requires an even number of active electrons")
    validate_active_space(molecule.active_orbitals, (molecule.active_electrons // 2,) * 2)
    elements = (
        "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
        "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba "
        "La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb "
        "Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh "
        "Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og"
    ).split()
    try:
        total_electrons = (
            sum(elements.index(symbol) + 1 for symbol in molecule.symbols) - molecule.charge
        )
    except ValueError as exc:
        raise ValueError("Unknown atomic symbol") from exc
    if total_electrons % 2 or molecule.active_electrons > total_electrons:
        raise ValueError("Active electrons must fit a closed-shell molecular electron count")
    if embedding_mode not in ("diagonal", "full_oneelectron"):
        raise ValueError("Unsupported embedding_mode")
    if (mm_charges is None) != (mm_coords is None):
        raise ValueError("MM charges and coordinates must be supplied together")
    if mm_charges is None:
        if embedding_mode == "full_oneelectron":
            raise ValueError("full_oneelectron requires MM inputs")
    else:
        charges = np.asarray(mm_charges)
        if charges.ndim != 1:
            raise ValueError("MM charges must be one-dimensional")
        _real_array(charges, charges.shape, "mm_charges")
        _real_array(mm_coords, (len(charges), 3), "mm_coords")


@dataclass(frozen=True)
class SQDConfig:
    """SQD run configuration; molecule retains its existing mutable semantics."""

    molecule: MoleculeConfig
    lucj: LUCJConfig = field(default_factory=LUCJConfig)
    reference: ReferenceConfig = field(default_factory=ReferenceConfig)
    embedding_mode: Literal["diagonal", "full_oneelectron"] = "diagonal"
    seed: int = 0
    allow_large: bool = False
    mode: RunMode = "full"

    def __post_init__(self) -> None:
        object.__setattr__(self, "molecule", deepcopy(self.molecule))

    def validate(self) -> SQDConfig:
        """Validate molecular and run inputs, returning this instance."""
        # MM values belong to the high-level input signature, not this configuration.
        validate_molecular_inputs(self.molecule)
        if self.embedding_mode not in ("diagonal", "full_oneelectron"):
            raise ValueError("Unsupported embedding_mode")
        _validate_pairs(self.lucj, self.molecule.active_orbitals)
        self.reference.validate()
        _validate_run(self.seed, self.allow_large, self.mode)
        return self

    def snapshot(self) -> JsonSnapshot:
        """Capture an independent immutable configuration at the time of the call.

        Returns:
            A recursively frozen mapping suitable for runtime provenance.

        Raises:
            ValueError: Configuration contains nonfinite or non-JSON metadata.
        """
        return _freeze_snapshot(self)


def _validate_run(seed: int, allow_large: bool, mode: str) -> None:
    _integer(seed, "seed")
    if not isinstance(allow_large, bool) or mode not in ("full", "reference_only"):
        raise ValueError("Invalid allow_large or run mode")


@dataclass(frozen=True)
class IntegralContext:
    """Immutable metadata describing the supplied fixed orbital Hamiltonian."""

    frame_id: str
    hamiltonian_id: str
    active_indices: tuple[int, ...]
    n_core_orbitals: int
    vacuum_core_constant: float
    delta_core_constant: float
    embedding_mode: Literal["vacuum", "diagonal", "full_oneelectron"]
    fixed_mo: bool
    two_electron_tensor_fixed: bool
    hf_reference_kind: Literal["canonical_rhf", "fixed_frame_determinant"]
    source: JsonSnapshot

    def __post_init__(self) -> None:
        object.__setattr__(self, "active_indices", tuple(self.active_indices))
        if not isinstance(self.source, Mapping):
            raise ValueError("Integral context source must be a mapping")
        object.__setattr__(self, "source", _freeze_snapshot(self.source))
        if not isinstance(self.frame_id, str) or not self.frame_id.strip():
            raise ValueError("frame_id must be nonempty")
        if (
            not isinstance(self.hamiltonian_id, str)
            or len(self.hamiltonian_id) != 64
            or any(c not in "0123456789abcdef" for c in self.hamiltonian_id)
        ):
            raise ValueError("hamiltonian_id must be a lowercase SHA256 digest")
        _integer(self.n_core_orbitals, "n_core_orbitals")
        for index in self.active_indices:
            _integer(index, "active index")
        if len(set(self.active_indices)) != len(self.active_indices):
            raise ValueError("Active indices must be unique")
        for name in ("vacuum_core_constant", "delta_core_constant"):
            _finite(getattr(self, name), name)
        if self.embedding_mode not in ("vacuum", "diagonal", "full_oneelectron"):
            raise ValueError("Invalid integral embedding mode")
        if not isinstance(self.fixed_mo, bool) or not isinstance(
            self.two_electron_tensor_fixed, bool
        ):
            raise ValueError("Frame flags must be booleans")
        if self.hf_reference_kind not in ("canonical_rhf", "fixed_frame_determinant"):
            raise ValueError("Invalid HF reference kind")
        if self.embedding_mode != "vacuum" and self.hf_reference_kind != "fixed_frame_determinant":
            raise ValueError("MM embedding requires a fixed-frame determinant reference")


@dataclass(frozen=True)
class CCSDSeed:
    """Copied read-only RCCSD amplitudes and declared solver provenance."""

    t1: np.ndarray
    t2: np.ndarray
    hf_energy: float
    ccsd_energy: float
    converged: bool
    residual_max_abs_ha: float
    frame_id: str
    hamiltonian_id: str
    solver: str
    solver_version: str

    def __post_init__(self) -> None:
        t1 = np.asarray(self.t1)
        if t1.ndim != 2:
            raise ValueError("t1 must be a matrix")
        nocc, nvirt = t1.shape
        for name, shape in (("t1", (nocc, nvirt)), ("t2", (nocc, nocc, nvirt, nvirt))):
            array = _real_array(getattr(self, name), shape, name).astype(float)
            # A bytes owner prevents callers from re-enabling WRITEABLE on the copy.
            frozen = np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(shape)
            object.__setattr__(self, name, frozen)
        for name in ("hf_energy", "ccsd_energy", "residual_max_abs_ha"):
            _finite(getattr(self, name), name)
        if self.residual_max_abs_ha < 0 or not isinstance(self.converged, bool):
            raise ValueError("Invalid CCSD residual or convergence flag")
        for name in ("frame_id", "hamiltonian_id", "solver", "solver_version"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be nonempty")


def validate_integral_inputs(
    h1: np.ndarray,
    h2: np.ndarray,
    e_core: float,
    *,
    norb: int,
    nelec: tuple[int, int],
    context: IntegralContext,
    seed_data: CCSDSeed | None,
    lucj: LUCJConfig | None = None,
    reference: ReferenceConfig | None = None,
    seed: int = 0,
    allow_large: bool = False,
    mode: RunMode = "full",
) -> None:
    """Validate integral structure, symmetry, and declared metadata consistency.

    This does not authenticate the Hamiltonian hash or independently recompute CCSD
    energies/residuals; the numerical receiving path must perform those checks too.

    Args:
        h1: Real one-electron integrals, shape ``(norb, norb)``.
        h2: Real chemist ERIs, shape ``(norb, norb, norb, norb)``.
        e_core: Total core constant in Hartree.
        norb: Number of active spatial orbitals.
        nelec: Balanced alpha and beta populations.
        context: Metadata for the supplied orbital frame and Hamiltonian.
        seed_data: Declared RCCSD seed, required in full mode.
        lucj: Optional ansatz and sampling configuration.
        reference: Optional reference budgets and solver configuration.
        seed: Nonnegative random seed.
        allow_large: Whether the resource policy may exceed its soft limit.
        mode: Full sampling or explicitly requested reference-only execution.

    Raises:
        ValueError: Shapes, symmetry, or metadata violate the supported domain.
        CCSDConvergenceError: Full mode is supplied an unconverged seed.
    """
    validate_active_space(norb, nelec)
    _validate_run(seed, allow_large, mode)
    _validate_pairs(lucj or LUCJConfig(), norb)
    (reference or ReferenceConfig()).validate()
    h1 = _real_array(h1, (norb, norb), "h1")
    h2 = _real_array(h2, (norb,) * 4, "h2")
    _finite(e_core, "e_core")
    for array, transpose in (
        (h1, h1.T),
        (h2, h2.transpose(1, 0, 2, 3)),
        (h2, h2.transpose(0, 1, 3, 2)),
        (h2, h2.transpose(2, 3, 0, 1)),
    ):
        if not np.allclose(array, transpose, atol=1e-12, rtol=0):
            raise ValueError("Integrals violate real chemist permutation symmetry")
    if not isinstance(context, IntegralContext) or len(context.active_indices) != norb:
        raise ValueError("IntegralContext must describe all active orbitals")
    if abs(e_core - context.vacuum_core_constant - context.delta_core_constant) > 1e-12:
        raise ValueError("e_core disagrees with the context constants")
    if seed_data is None:
        if mode == "full":
            raise ValueError("Full mode requires CCSDSeed")
        return
    if not isinstance(seed_data, CCSDSeed):
        raise ValueError("seed_data must be a CCSDSeed")
    if seed_data.t1.shape != (nelec[0], norb - nelec[0]):
        raise ValueError("CCSD amplitudes do not match the active space")
    if seed_data.frame_id != context.frame_id or seed_data.hamiltonian_id != context.hamiltonian_id:
        raise ProvenanceMismatchError("CCSD seed frame or Hamiltonian mismatch")
    if mode == "full" and (not seed_data.converged or seed_data.residual_max_abs_ha > 1e-7):
        raise CCSDConvergenceError("Full mode requires a converged CCSD seed")
