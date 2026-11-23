# Copyright (c) 2026 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT

"""Monte Carlo solvation driven by sample-based quantum diagonalization (SQD).

Each Monte Carlo trial calls the SQD solver with the current MM point charges.
The Metropolis acceptance energy is ``E_sqd + E_mm(sol-sol) + E_LJ(QM-MM)``.
This entry point is independent of the QPE workflow in ``run_solvation``.
The SQD backend loads only when a step runs, so importing this module needs
no optional extra.
"""

from __future__ import annotations

import os
import platform
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from q2m3.constants import BANNED_TERMS, BOLTZMANN_CONSTANT

from .energy import StepResult
from .mc_loop import MCResult, create_mc_loop
from .solvent import (
    TIP3P_WATER,
    SolventMolecule,
    compute_mm_energy,
    get_mm_embedding_data,
    initialize_solvent_ring,
    molecules_to_state_array,
    qm_mm_lj_energy,
    state_array_to_molecules,
    validate_qm_mm_lj_elements,
)

if TYPE_CHECKING:
    import q2m3.molecule

FULL_SUBSPACE_STATEMENT = (
    "In this domain SQD equals exact diagonalization; the sampler adds no information "
    "beyond CASCI."
)
"""Summary statement for runs where every SQD subspace spans the full CI space."""

THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
"""Environment variables that must equal ``"1"`` before an SQD run."""

ENERGY_MODEL = "E_sqd + E_mm(sol-sol) + E_LJ(QM-MM)"
LJ_MODEL = "QM-MM Lennard-Jones, Lorentz-Berthelot rules, QM_MM_LJ_PARAMETERS + TIP3P sites"
FULL_SUBSPACE_DOMAIN = "full_subspace_exact_diagonalization"
PARTIAL_SUBSPACE_DOMAIN = "partial_subspace"
_EMBEDDING_MODES = ("diagonal", "full_oneelectron")


class SubspaceDomainError(ValueError):
    """Raised when the SQD subspace is smaller than the full CI space.

    A Monte Carlo step raises this error unless
    ``SQDSolvationConfig.allow_partial_subspace`` is True.
    """


@dataclass(frozen=True)
class SQDSolvationConfig:
    """Settings for an SQD-driven Monte Carlo solvation run.

    The step and temperature defaults match ``SolvationConfig``. The object
    satisfies the settings protocol of ``create_mc_loop``.

    Attributes:
        molecule: QM solute (``q2m3.molecule.MoleculeConfig``). Coordinates in Angstrom.
        n_waters: Number of TIP3P water molecules.
        n_mc_steps: Number of Monte Carlo steps after the initial evaluation.
        temperature: Temperature in Kelvin.
        translation_step: Maximum water translation per step in Angstrom.
        rotation_step: Maximum water rotation per step in radians.
        initial_water_distance: Radius of the initial water ring in Angstrom.
        random_seed: Seed for the initial ring orientations and the Monte Carlo RNG.
        embedding_mode: MM embedding passed to the SQD solver.
        sqd_seed: SQD sampling seed. Every step uses the same value.
        lucj: LUCJ ansatz settings (``q2m3.sqd.LUCJConfig``), or None for defaults.
        reference: Classical reference settings (``q2m3.sqd.ReferenceConfig``),
            or None for defaults.
        allow_partial_subspace: If False, a step whose SQD subspace is smaller
            than the full CI space raises ``SubspaceDomainError``.
    """

    molecule: q2m3.molecule.MoleculeConfig
    n_waters: int = 8
    n_mc_steps: int = 500
    temperature: float = 300.0
    translation_step: float = 0.3
    rotation_step: float = 0.2618
    initial_water_distance: float = 3.5
    random_seed: int = 42
    embedding_mode: Literal["diagonal", "full_oneelectron"] = "full_oneelectron"
    sqd_seed: int = 0
    lucj: LUCJConfig | None = None  # noqa: F821
    reference: ReferenceConfig | None = None  # noqa: F821
    allow_partial_subspace: bool = False

    @property
    def kt(self) -> float:
        """Thermal energy k_B * T in Hartree."""
        return BOLTZMANN_CONSTANT * self.temperature

    @property
    def acceptance_uses_hf_reference(self) -> bool:
        """Always False: the acceptance energy uses the SQD energy."""
        return False

    def validate(self) -> SQDSolvationConfig:
        """Check the settings and the runtime environment.

        Returns:
            This configuration.

        Raises:
            ValueError: If a count, step, distance or temperature is not positive,
                if ``embedding_mode`` is unknown, if an element has no QM-MM
                Lennard-Jones parameters, if no solute atom has epsilon > 0,
                if the platform is not Linux x86_64,
                or if a thread variable in ``THREAD_VARIABLES`` is not ``"1"``.
        """
        for name in (
            "n_waters",
            "n_mc_steps",
            "temperature",
            "translation_step",
            "rotation_step",
            "initial_water_distance",
        ):
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)!r}")
        if self.embedding_mode not in _EMBEDDING_MODES:
            raise ValueError(
                f"embedding_mode must be one of {_EMBEDDING_MODES}, got {self.embedding_mode!r}"
            )
        validate_qm_mm_lj_elements(self.molecule.symbols)
        if platform.system() != "Linux" or platform.machine() != "x86_64":
            raise ValueError("SQD Monte Carlo solvation requires Linux x86_64")
        for name in THREAD_VARIABLES:
            if os.environ.get(name) != "1":
                raise ValueError(f"SQD requires {name}=1 before interpreter start")
        return self


@dataclass(frozen=True)
class SQDMCResult:
    """Result of ``run_sqd_solvation``.

    Attributes:
        mc: Monte Carlo result with the per-step decisions and step results.
        initial_step: Step result of the initial solvent configuration.
        sqd_results: SQD result (``q2m3.sqd.SQDResult``) of every Monte Carlo trial.
        accepted: Metropolis decision of every step, shape (n_mc_steps,).
        min_qm_mm_A: Nearest QM-MM atom distance in Angstrom of the configuration
            after each step's decision, shape (n_mc_steps,).
        gap_warnings: One entry per pair of adjacent accepted states whose gap
            ``sqd_energy - baseline_energy`` changes by more than kT. Filled only
            when partial subspaces are allowed.
        provenance: Domain, energy model, seeds, embedding mode and thread settings.
    """

    mc: MCResult
    initial_step: StepResult
    sqd_results: tuple[Any, ...]
    accepted: np.ndarray
    min_qm_mm_A: np.ndarray
    gap_warnings: tuple[dict[str, float], ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)


def create_sqd_step_callback(
    config: SQDSolvationConfig,
    *,
    solver: Callable[..., Any] | None = None,
) -> Callable[[np.ndarray, np.ndarray], StepResult]:
    """Create a Monte Carlo step callback that evaluates the QM energy with SQD.

    Args:
        config: Run settings.
        solver: Callable with the signature of ``q2m3.sqd.run_sqd``. None loads
            ``run_sqd`` from the ``sqd`` extra.

    Returns:
        Callback ``(solvent_states, qm_coords) -> StepResult``. ``qm_coords`` can be
        flat or of shape (n_qm, 3), in Angstrom. Exceptions from the solver
        propagate unchanged.

    Raises:
        ImportError: If ``solver`` is None and the ``sqd`` extra is not installed.
    """
    solve: Callable[..., Any]
    if solver is None:
        from q2m3.sqd import run_sqd as solve
    else:
        solve = solver

    molecule = config.molecule
    symbols = list(molecule.symbols)

    def step(solvent_states: np.ndarray, qm_coords: np.ndarray) -> StepResult:
        """Evaluate one solvent configuration.

        Raises:
            RuntimeError: If the SQD run did not complete with a finite energy.
            SubspaceDomainError: If the subspace is partial and not allowed.
        """
        start = time.perf_counter()
        qm = np.asarray(qm_coords, dtype=np.float64).reshape(-1, 3)
        molecules = state_array_to_molecules(TIP3P_WATER, solvent_states)
        mm_coords, mm_charges = get_mm_embedding_data(molecules)

        solver_start = time.perf_counter()
        result = solve(
            symbols,
            qm,
            active_electrons=molecule.active_electrons,
            active_orbitals=molecule.active_orbitals,
            charge=molecule.charge,
            basis=molecule.basis,
            lucj=config.lucj,
            reference=config.reference,
            mm_charges=mm_charges,
            mm_coords=mm_coords,
            embedding_mode=config.embedding_mode,
            seed=config.sqd_seed,
            verbose=False,
        )
        solver_time = time.perf_counter() - solver_start

        energy = result.sqd_energy
        if result.status != "completed" or energy is None or not np.isfinite(energy):
            raise RuntimeError(
                f"SQD step did not complete: status={result.status!r}, sqd_energy={energy!r}"
            )
        if not config.allow_partial_subspace and result.subspace_dim != result.full_ci_dim:
            raise SubspaceDomainError(
                f"SQD subspace dimension {result.subspace_dim} is smaller than the full CI "
                f"dimension {result.full_ci_dim}. Set allow_partial_subspace=True to accept it."
            )

        e_mm_sol_sol = compute_mm_energy(molecules)
        e_qmmm_vdw = qm_mm_lj_energy(symbols, qm, molecules)
        return StepResult(
            e_qpe=float(energy),
            e_mm_sol_sol=e_mm_sol_sol,
            e_hf_ref=result.hf_energy,
            callback_time=time.perf_counter() - start,
            qpe_time=solver_time,
            quantum_method="sqd",
            e_qmmm_vdw=e_qmmm_vdw,
            sqd_result=result,
        )

    return step


def _min_qm_mm_distance(qm_coords: np.ndarray, molecules: list[SolventMolecule]) -> float:
    mm_coords, _ = get_mm_embedding_data(molecules)
    return float(np.linalg.norm(qm_coords[:, None, :] - mm_coords[None, :, :], axis=-1).min())


def _gap(result: Any) -> float:
    return result.sqd_energy - result.baseline_energy


def _gap_warnings(
    initial: Any, sqd_results: tuple[Any, ...], accepted: np.ndarray, kt: float
) -> list[dict[str, float]]:
    warnings = []
    previous = _gap(initial)
    for index, (is_accepted, result) in enumerate(zip(accepted, sqd_results, strict=True)):
        if not is_accepted:
            continue
        gap = _gap(result)
        if abs(gap - previous) > kt:
            warnings.append({"step": index, "previous_gap_ha": previous, "gap_ha": gap})
        previous = gap
    return warnings


def run_sqd_solvation(
    config: SQDSolvationConfig,
    *,
    solver: Callable[..., Any] | None = None,
) -> SQDMCResult:
    """Run Monte Carlo solvation with SQD energies of the QM region.

    The initial water ring reseeds the global ``np.random`` state with
    ``config.random_seed``. The Monte Carlo loop uses its own generator.

    Args:
        config: Run settings.
        solver: Callable with the signature of ``q2m3.sqd.run_sqd``. None loads
            ``run_sqd`` from the ``sqd`` extra.

    Returns:
        Per-step SQD results, Monte Carlo decisions and run provenance.

    Raises:
        ValueError: If ``config.validate()`` fails.
        SubspaceDomainError: If a step has a partial subspace that is not allowed.
            The initial configuration is checked before the first Monte Carlo step.
        RuntimeError: If an SQD step does not complete with a finite energy.
        ImportError: If ``solver`` is None and the ``sqd`` extra is not installed.
    """
    config.validate()
    qm_coords = np.asarray(config.molecule.coords_array, dtype=np.float64)
    molecules = initialize_solvent_ring(
        TIP3P_WATER,
        config.n_waters,
        qm_coords.mean(axis=0),
        config.initial_water_distance,
        config.random_seed,
    )
    initial_states = molecules_to_state_array(molecules)
    step = create_sqd_step_callback(config, solver=solver)

    initial_step = step(initial_states, qm_coords.flatten())
    mc = create_mc_loop(config, step)(
        initial_states,
        qm_coords.flatten(),
        config.random_seed,
        initial_step.acceptance_energy(config.acceptance_uses_hf_reference),
    )

    sqd_results = tuple(r.sqd_result for r in mc.step_results)
    min_qm_mm = np.array(
        [
            _min_qm_mm_distance(qm_coords, state_array_to_molecules(TIP3P_WATER, states))
            for states in mc.trajectory_solvent_states
        ]
    )
    all_results: tuple[Any, ...] = (initial_step.sqd_result, *sqd_results)
    full = all(r.subspace_dim == r.full_ci_dim for r in all_results)
    warnings = []
    if config.allow_partial_subspace:
        warnings = _gap_warnings(initial_step.sqd_result, sqd_results, mc.accepted, config.kt)

    provenance = {
        "domain": FULL_SUBSPACE_DOMAIN if full else PARTIAL_SUBSPACE_DOMAIN,
        "energy_model": ENERGY_MODEL,
        "lj_model": LJ_MODEL,
        "sqd_seed": config.sqd_seed,
        "random_seed": config.random_seed,
        "embedding_mode": config.embedding_mode,
        **{name: os.environ.get(name) for name in THREAD_VARIABLES},
    }
    return SQDMCResult(
        mc=mc,
        initial_step=initial_step,
        sqd_results=sqd_results,
        accepted=mc.accepted,
        min_qm_mm_A=min_qm_mm,
        gap_warnings=tuple(warnings),
        provenance=provenance,
    )


def _format_delta(delta: float | None) -> str:
    return "n/a" if delta is None else f"{delta:.2f}"


def render_sqd_mc_summary(result: SQDMCResult) -> str:
    """Render a Markdown summary of an SQD Monte Carlo run.

    The summary lists the acceptance rate, the step count, the smallest nearest
    QM-MM distance of the configurations kept after each step, and per step the SQD error
    against the classical baseline with the baseline tier and method.

    Args:
        result: Output of ``run_sqd_solvation``.

    Returns:
        Markdown text.

    Raises:
        ValueError: If the text contains a term from ``q2m3.constants.BANNED_TERMS``
            (case-insensitive).
    """
    accepted = np.asarray(result.accepted, dtype=bool)
    min_distance = float(np.min(result.min_qm_mm_A)) if len(result.min_qm_mm_A) else float("nan")
    lines = [
        "## SQD Monte Carlo solvation",
        "",
        f"- Steps: {len(accepted)}",
        f"- Acceptance rate: {accepted.mean() if len(accepted) else 0.0:.3f}",
        f"- Minimum nearest QM-MM distance: {min_distance:.3f} Angstrom",
        f"- Domain: {result.provenance.get('domain')}",
        f"- Gap warnings: {len(result.gap_warnings)}",
        "",
        "| Step | Accepted | delta_mHa | baseline_tier | baseline_method |",
        "|------|----------|-----------|---------------|-----------------|",
    ]
    for index, (is_accepted, sqd) in enumerate(zip(accepted, result.sqd_results, strict=True)):
        lines.append(
            f"| {index} | {'yes' if is_accepted else 'no'} | {_format_delta(sqd.delta_mHa)} "
            f"| {sqd.baseline_tier} | {sqd.baseline_method} |"
        )
    if result.provenance.get("domain") == FULL_SUBSPACE_DOMAIN:
        lines += ["", FULL_SUBSPACE_STATEMENT]
    text = "\n".join(lines) + "\n"

    folded = text.casefold()
    found = [term for term in BANNED_TERMS if term.casefold() in folded]
    if found:
        raise ValueError(f"SQD Monte Carlo summary contains banned terms: {found}")
    return text
