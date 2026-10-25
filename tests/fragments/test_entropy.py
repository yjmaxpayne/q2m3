"""Orbital-entropy diagnostics of an active space from a block2 DMRG run.

A tier is a run of sweeps with one bond dimension; its energy is the energy after its last
sweep. A run whose last two tier energies differ by more than 1 mHa is unconverged.
Unmarked tests feed synthetic sweep records, entropies and orbitals to the helpers and need
no block2. Tests marked ``dmrg`` run block2 and skip when it is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest
from pyscf import fci

from q2m3 import _lazy
from q2m3.fragments.active_space import select_active_spaces
from q2m3.fragments.counterpoise import counterpoise
from q2m3.fragments.diagnostics import entropy
from q2m3.fragments.diagnostics.entropy import (
    DMRGConvergence,
    assign_orbitals,
    entropy_diagnostics,
    mutual_information,
)
from q2m3.fragments.integrals import build_integrals
from q2m3.fragments.molecule import Molecule
from tests.fragments.conftest import AVAS_SETTINGS, WATER_DIMER

DIMER_FRAGMENTS = ((0, 1, 2), (3, 4, 5))
LABELS, THRESHOLD = AVAS_SETTINGS["S6"]
# Water-dimer DMRG energies (Ha) of the prototype runs at M=100 and M=250
# (cc-pVDZ, 26 active orbitals); they differ by 9.5 mHa.
PROTOTYPE_M100_HA = -152.41547634385688
PROTOTYPE_M250_HA = -152.4249797312157
# A (12e, 8o) space is small enough for block2 at M=100 to reproduce CASCI.
CASCI_TOLERANCE_HA = 1e-6

requires_block2 = pytest.mark.skipif(
    not _lazy.extra_available("dmrg"), reason="the optional 'dmrg' backend is not installed"
)


def _molecule() -> Molecule:
    lines = [str(len(WATER_DIMER)), ""] + [f"{s} {x!r} {y!r} {z!r}" for s, (x, y, z) in WATER_DIMER]
    return Molecule.from_xyz("\n".join(lines) + "\n", fragments=DIMER_FRAGMENTS)


def test_prototype_schedule_is_unconverged():
    run = DMRGConvergence((100, 250), (PROTOTYPE_M100_HA, PROTOTYPE_M250_HA))
    assert run.status == "unconverged"
    assert run.last_tier_difference_ha == pytest.approx(9.503e-3, abs=1e-6)


@pytest.mark.parametrize("difference_ha, status", [(0.9e-3, "converged"), (1.1e-3, "unconverged")])
def test_threshold_separates_converged_from_unconverged(difference_ha, status):
    run = DMRGConvergence((100, 250), (-152.0, -152.0 - difference_ha))
    assert run.status == status


def test_difference_equal_to_the_threshold_is_converged():
    run = DMRGConvergence((100, 250), (0.0, -1e-3))
    assert run.last_tier_difference_ha == 1e-3
    assert run.status == "converged"


def test_tier_energy_is_the_energy_after_its_last_sweep():
    run = DMRGConvergence((50, 50, 100, 100), (-1.0, -1.1, -1.1002, -1.1004))
    assert run.tier_bond_dims == (50, 100)
    assert run.tier_energies == (-1.1, -1.1004)
    assert run.status == "converged"


def test_verdict_uses_the_last_two_tiers():
    run = DMRGConvergence((50, 100, 250), (-1.0, -1.01, -1.0105))
    assert run.last_tier_difference_ha == pytest.approx(0.5e-3, abs=1e-12)
    assert run.status == "converged"


def test_single_bond_dimension_is_rejected():
    with pytest.raises(ValueError, match="at least two bond dimensions"):
        DMRGConvergence((100, 100), (-1.0, -1.1))


def test_mutual_information_halves_the_entropy_sum_and_zeroes_the_diagonal():
    s1 = np.array([0.1, 0.2, 0.4])
    s2 = np.array([[0.0, 0.25, 0.45], [0.25, 0.0, 0.5], [0.45, 0.5, 0.0]])
    expected = np.array([[0.0, 0.025, 0.025], [0.025, 0.0, 0.05], [0.025, 0.05, 0.0]])
    np.testing.assert_allclose(mutual_information(s1, s2), expected, rtol=0, atol=1e-15)


def test_orbitals_are_assigned_by_lowdin_population_on_fragment_a():
    molecule = _molecule()
    subsystem = counterpoise(molecule, basis="sto-3g")["AB"]
    mol = subsystem.mol
    w, v = np.linalg.eigh(mol.intor_symmetric("int1e_ovlp"))
    inverse_sqrt = (v / np.sqrt(w)) @ v.T
    first_ao = mol.aoslice_by_atom()[:, 2]
    # Each column becomes the unit vector of one AO after Lowdin orthogonalization.
    orbitals = inverse_sqrt[:, [first_ao[0], first_ao[3]]]
    weights, fragments = assign_orbitals(molecule, subsystem, orbitals)
    np.testing.assert_allclose(weights, [1.0, 0.0], rtol=0, atol=1e-12)
    assert fragments == ("A", "B")


def test_missing_backend_raises_with_the_install_command(monkeypatch):
    monkeypatch.setattr(entropy, "extra_available", lambda extra: extra != "dmrg")
    with pytest.raises(ImportError, match="uv sync --extra dmrg"):
        entropy_diagnostics(None, None, None, None)


@pytest.fixture(scope="module")
def dimer_space():
    molecule = _molecule()
    subsystem = counterpoise(molecule, basis="cc-pvdz")["AB"]
    record, mo = select_active_spaces({"AB": subsystem}, LABELS, THRESHOLD)["AB"]
    integrals = build_integrals(
        molecule, subsystem, record, mo, avas_labels=LABELS, avas_threshold=THRESHOLD
    )
    return molecule, subsystem, integrals, mo


@pytest.mark.dmrg
@requires_block2
def test_water_dimer_diagnostics_match_casci(dimer_space):
    molecule, subsystem, integrals, mo = dimer_space
    result = entropy_diagnostics(molecule, subsystem, integrals, mo, bond_dims=(50, 100))
    ncas = integrals.record.ncas
    assert (integrals.record.nelecas, ncas) == (12, 8)
    assert result.single_orbital_entropy.shape == (ncas,)
    assert np.all(
        (result.single_orbital_entropy >= 0) & (result.single_orbital_entropy <= np.log(4))
    )
    mi = result.mutual_information
    assert mi.shape == (ncas, ncas)
    # block2 returns two-orbital entropies that are symmetric to about 1e-15 only.
    np.testing.assert_allclose(mi, mi.T, rtol=0, atol=1e-12)
    assert np.all(np.diag(mi) == 0.0) and mi.min() > -1e-12
    assert result.fragment_weights.shape == (ncas,)
    assert result.orbital_fragments == tuple(
        "A" if w >= 0.5 else "B" for w in result.fragment_weights
    )
    assert result.assignment_rule == entropy.ASSIGNMENT_RULE
    assert result.convergence.sweep_bond_dims == (50,) * 4 + (100,) * 4
    assert result.convergence.status == "converged"
    e_casci = fci.direct_spin1.kernel(integrals.h1, integrals.h2, ncas, integrals.nelec)[0]
    assert result.convergence.tier_energies[-1] == pytest.approx(
        e_casci + integrals.e_core, abs=CASCI_TOLERANCE_HA
    )


@pytest.mark.dmrg
@requires_block2
def test_low_bond_dimensions_are_flagged_unconverged(dimer_space):
    result = entropy_diagnostics(*dimer_space, bond_dims=(4, 8), sweeps_per_tier=2)
    assert result.convergence.sweep_bond_dims == (4, 4, 8, 8)
    assert result.convergence.status == "unconverged"


@pytest.mark.dmrg
@requires_block2
def test_every_requested_sweep_runs(dimer_space):
    # block2 stops early once a noise-free sweep at the last bond dimension changes the
    # energy by less than its tolerance; repeated last bond dimensions allow that stop.
    result = entropy_diagnostics(*dimer_space, bond_dims=(50, 100, 100, 100), sweeps_per_tier=1)
    assert result.convergence.sweep_bond_dims == (50, 100, 100, 100)


@pytest.mark.dmrg
@requires_block2
def test_thread_count_reaches_block2(dimer_space, monkeypatch):
    from pyblock2.driver import core

    seen = []

    class RecordingDriver(core.DMRGDriver):
        def __init__(self, *args, **kwargs):
            seen.append(kwargs.get("n_threads"))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(core, "DMRGDriver", RecordingDriver)
    entropy_diagnostics(*dimer_space, bond_dims=(4, 8), sweeps_per_tier=1, n_threads=2)
    assert seen == [2]
