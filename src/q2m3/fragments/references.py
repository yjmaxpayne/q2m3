"""Full-space reference energies of the counterpoise subsystems.

The CCSD(T) reference runs only when its predicted peak RSS and wall time fit the budget.
The cost model is the POC-C fit for PySCF 2.14 RHF with frozen-core CCSD(T) in cc-pVDZ:
memory in MB of 1e6 bytes, wall time in seconds at 8 threads.
"""

from __future__ import annotations

import dataclasses
import resource
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol

from pyscf import cc, lib, mp
from pyscf.data.elements import chemcore

from q2m3.fragments.counterpoise import Subsystem
from q2m3.fragments.errors import FragmentWorkflowError

MODEL_ID = "pyscf-2.14.0-rccsd_t-ccpvdz-v1"
BASE_MB = 144.9
WALL_COEF = (9.5105e-01, 3.2717e-11, 0.0, 2.9930e-08)
CALIBRATED_NAO = 157
CALIBRATED_THREADS = 8

_W = 8 / 1e6  # MB per float64
_DIIS_INCORE_SIZE = 1e7  # pyscf.lib.diis.INCORE_SIZE (elements)
_DIIS_VECS = 13  # diis_space=6: 6 x-vectors, 6 error vectors and xprev
_N_AMP_WORK = 6  # t2, t2new, tau, two W(ovov) intermediates, t2 copy in update_amps


@dataclasses.dataclass(frozen=True)
class CCSDTBudget:
    """Upper limits of one CCSD(T) stage: peak RSS (MB) and summed wall time (s)."""

    rss_mb: float = 20000.0
    wall_s: float = 7200.0


@dataclasses.dataclass(frozen=True)
class CostEstimate:
    """Predicted peak RSS (MB) and wall time (s) of one frozen-core CCSD(T) run."""

    rss_mb: float
    wall_s: float
    cc_incore: bool
    model_id: str
    extrapolated: bool


class CostModel(Protocol):
    def __call__(
        self, nao: int, nocc: int, nvir: int, nfrozen: int, *, max_memory_mb: float
    ) -> CostEstimate: ...


@dataclasses.dataclass(frozen=True)
class CCSDTStage:
    """Outcome of the budget gate and, when it passes, of the CCSD(T) runs.

    ``energies`` maps each subsystem to its CCSD(T) total energy (Ha), or to ``None`` when
    the gate rejects the stage. ``reason`` is the rejection reason. ``cost`` is the
    provenance record of the estimates, the budget and the measured cost.
    """

    ran: bool
    energies: dict[str, float | None]
    reason: str | None
    cost: dict[str, Any]
    warnings: tuple[str, ...]


def frozen_core_mp2(subsystem: Subsystem) -> tuple[float, int]:
    """Return the full-space MP2 total energy (Ha) and the number of frozen orbitals.

    The frozen orbitals are the chemical core of the atoms with a nucleus. Ghost atoms
    add no core orbitals, so a monomer in the dimer basis freezes only its own core.
    """
    nfrozen = chemcore(subsystem.mol)
    solver = mp.MP2(subsystem.scf, frozen=nfrozen)
    solver.kernel()
    return float(solver.e_tot), nfrozen


def frozen_core_ccsd_t(subsystem: Subsystem, nfrozen: int, *, max_memory_mb: float) -> float:
    """Return the frozen-core CCSD(T) total energy (Ha) of ``subsystem``.

    Raises:
        FragmentWorkflowError: The CCSD iterations do not converge.
    """
    solver = cc.CCSD(subsystem.scf, frozen=nfrozen)
    solver.max_memory = max_memory_mb
    eris = solver.ao2mo()
    solver.kernel(eris=eris)
    if not solver.converged:
        raise FragmentWorkflowError(f"CCSD of subsystem {subsystem.name} did not converge")
    return float(solver.e_tot + solver.ccsd_t(eris=eris))


def _pair(n: int) -> int:
    return n * (n + 1) // 2


def estimate_ccsd_t_cost(
    nao: int, nocc: int, nvir: int, nfrozen: int, *, max_memory_mb: float
) -> CostEstimate:
    """Predict the peak RSS and wall time of RHF plus frozen-core CCSD(T).

    ``nocc`` counts the active occupied orbitals, after the frozen core. The RSS is
    ``BASE_MB`` plus the largest of the SCF, CCSD and (T) stage peaks. The CCSD integrals
    stay in memory only when they fit ``max_memory_mb``. Otherwise PySCF sizes its buffers
    from ``max_memory_mb``, so the RSS rises to that value. ``nfrozen`` is part of the
    cost-model interface; this model does not use it.
    """
    o, v = nocc, nvir
    nmo = o + v
    npair = _pair(nao)
    scf_incore = nao**4 / 1e6 + BASE_MB < 0.95 * max_memory_mb
    ao_eri = _pair(npair) * _W if scf_incore else 0.0
    vp = _pair(v)
    # MO blocks oooo, ovoo, ovov + oovv + ovvo, ovvv and vvvv.
    mo_blocks = (o**4 + o**3 * v + 3 * o**2 * v**2 + o * v * vp + vp**2) * _W
    mem_incore = (max(npair**2, nmo**4) + _pair(nmo) ** 2) * _W
    cc_incore = scf_incore and mem_incore + BASE_MB + ao_eri < max_memory_mb
    amp = o * o * v * v * _W
    diis = _DIIS_VECS * amp if o * o * v * v + o * v < _DIIS_INCORE_SIZE else 0.0
    # Block buffers of update_amps and _add_ovvv_ (block size up to nvir).
    cc_buf = (
        max(
            v * (7 * o * o * v + o**3 + o * v * v), v * (3 * o * v * v + o * o * v + 2) + o * o * vp
        )
        * _W
    )
    if cc_incore:
        held = ao_eri + mo_blocks
        build = max(
            ao_eri + npair * _pair(nmo) * _W + _pair(nmo) ** 2 * _W,
            ao_eri + _pair(nmo) ** 2 * _W + mo_blocks,
        )
        avail = max(0.0, max_memory_mb - BASE_MB - held)
        ccsd = max(build, held + _N_AMP_WORK * amp + diis + min(cc_buf, 0.9 * avail))
    else:
        held = ao_eri
        avail = max(0.0, max_memory_mb - BASE_MB - held)
        ccsd = max(held + _N_AMP_WORK * amp + diis, max_memory_mb - BASE_MB)
    vvop = v * v * o * nmo * _W
    triples = held + 2 * amp + o**3 * v * _W + min(vvop, 0.5 * avail)
    terms = (1.0, o**3 * v**4, o**2 * v**4, float(nao) ** 4)
    return CostEstimate(
        rss_mb=BASE_MB + max(ao_eri, ccsd, triples),
        wall_s=sum(c * t for c, t in zip(WALL_COEF, terms, strict=True)),
        cc_incore=cc_incore,
        model_id=MODEL_ID,
        extrapolated=nao > CALIBRATED_NAO,
    )


def _proc_mb(path: str, key: str) -> float:
    for line in Path(path).read_text().splitlines():
        if line.startswith(key):
            return int(line.split()[1]) * 1024 / 1e6
    raise RuntimeError(f"{path} has no {key} line")


def mem_available_mb() -> float:
    """Return ``MemAvailable`` of ``/proc/meminfo`` in MB."""
    return _proc_mb("/proc/meminfo", "MemAvailable:")


def run_ccsd_t_stage(
    subsystems: Mapping[str, Subsystem],
    nfrozen: Mapping[str, int],
    budget: CCSDTBudget,
    *,
    cost_model: CostModel = estimate_ccsd_t_cost,
    mem_available_mb: Callable[[], float] = mem_available_mb,
    solver: Callable[..., float] = frozen_core_ccsd_t,
) -> CCSDTStage:
    """Predict the CCSD(T) cost of every subsystem and run them when the budget allows.

    The effective RSS budget is ``min(budget.rss_mb, MemAvailable)``. PySCF gets that
    budget minus ``BASE_MB`` as ``max_memory``. The stage runs when the largest RSS
    prediction and the summed wall-time prediction are at most their budgets.

    The measured ``peak_rss_mb`` is the process peak after each subsystem. It includes
    ``rss_before_mb``, the resident set of the process before the stage, which the
    standalone-process cost model does not include.
    """
    available = mem_available_mb()
    effective = min(budget.rss_mb, available)
    max_memory = effective - BASE_MB
    estimates = {}
    for name, sub in subsystems.items():
        nocc_all = sub.mol.nelectron // 2
        nmo = sub.scf.mo_coeff.shape[1]
        estimates[name] = cost_model(
            sub.mol.nao,
            nocc_all - nfrozen[name],
            nmo - nocc_all,
            nfrozen[name],
            max_memory_mb=max_memory,
        )
    predicted = {
        "rss_mb": max(e.rss_mb for e in estimates.values()),
        "wall_s": sum(e.wall_s for e in estimates.values()),
    }
    limits = {"rss_mb": effective, "wall_s": budget.wall_s}
    exceeded = [
        f"predicted {key} {predicted[key]:.1f} > budget {limits[key]:.1f}"
        for key in ("rss_mb", "wall_s")
        if predicted[key] > limits[key]
    ]
    threads = lib.num_threads()
    cost: dict[str, Any] = {
        "model_id": next(iter(estimates.values())).model_id,
        "budget": dataclasses.asdict(budget),
        "mem_available_mb": available,
        "effective_rss_mb": effective,
        "max_memory_mb": max_memory,
        "estimates": {name: dataclasses.asdict(e) for name, e in estimates.items()},
        "predicted": predicted,
        "ran": not exceeded,
        "threads": threads,
        "rss_before_mb": _proc_mb("/proc/self/status", "VmRSS:"),
        "measured": {},
    }
    warnings: tuple[str, ...] = ()
    if threads != CALIBRATED_THREADS:
        warnings = (
            f"CCSD(T) wall-time model is calibrated at {CALIBRATED_THREADS} threads; "
            f"this run uses {threads}",
        )
    energies: dict[str, float | None] = dict.fromkeys(subsystems)
    if exceeded:
        return CCSDTStage(False, energies, "; ".join(exceeded), cost, warnings)
    for name, sub in subsystems.items():
        start = time.perf_counter()
        energies[name] = solver(sub, nfrozen[name], max_memory_mb=max_memory)
        cost["measured"][name] = {
            "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / 1e6,
            "wall_s": time.perf_counter() - start,
        }
    return CCSDTStage(True, energies, None, cost, warnings)
