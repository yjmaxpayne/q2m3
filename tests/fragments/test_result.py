"""Fragment result records: interaction energies are rechecked from the subsystem energies.

The records are built directly, so these checks need no SCF run.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from q2m3.fragments.result import (
    SCHEMA_VERSION,
    ErrorDecomposition,
    FragmentInteractionResult,
    SubsystemResult,
)

# Synthetic energies in Ha; only their differences matter here.
ENERGIES = {"AB": (-152.0625, -152.4687), "A": (-76.0269, -76.2290), "B": (-76.0297, -76.2333)}
ERROR_TERMS = ("algorithm", "active_space", "basis", "model")
NO_SQD = "mode='classical_only' does not run the SQD stage"


def _subsystem(name: str, e_hf: float | None, e_mp2: float | None) -> SubsystemResult:
    reasons = {key: NO_SQD for key in ("e_casci", "e_sqd", "sqd")}
    reasons |= {
        "e_ccsd_t": "not computed",
        "active_space": "not resolved",
        "resources.qpe": "not computed",
        "resources.sqd": "not computed",
    }
    return SubsystemResult(
        name=name,
        real_atoms=(0, 1, 2),
        ghost_atoms=(3, 4, 5),
        n_ao=48,
        e_hf=e_hf,
        e_mp2=e_mp2,
        e_ccsd_t=None,
        e_casci=None,
        e_sqd=None,
        active_space=None,
        sqd=None,
        resources={},
        null_reasons=reasons,
    )


# A classical-only result has no CCSD(T) cost record.
CLASSICAL_PROVENANCE_KEYS = (
    "geometry_sha256",
    "basis",
    "cart",
    "avas_labels",
    "avas_threshold",
    "cos_threshold",
    "seed",
    "scf_conv_tol",
    "frozen_core",
    "versions",
    "threads",
)


def _result(**changes) -> FragmentInteractionResult:
    subsystems = {name: _subsystem(name, *values) for name, values in ENERGIES.items()}
    ab, a, b = (ENERGIES[name] for name in ("AB", "A", "B"))
    delta_e = {
        "hf": ab[0] - a[0] - b[0],
        "mp2": ab[1] - a[1] - b[1],
        "casci": None,
        "sqd": None,
        "ccsd_t": None,
    }
    reasons = {
        f"delta_e.{layer}": "subsystem energy is null" for layer in ("casci", "sqd", "ccsd_t")
    }
    reasons |= {f"errors.{term}": "no reference" for term in ERROR_TERMS}
    reasons |= {"correspondence": "not resolved"}
    fields = {
        "fragment_id": "water-dimer",
        "mode": "classical_only",
        "basis": "cc-pvdz",
        "subsystems": subsystems,
        "delta_e": delta_e,
        "references": (),
        "errors": ErrorDecomposition(),
        "correspondence": None,
        "provenance": dict.fromkeys(CLASSICAL_PROVENANCE_KEYS, "recorded"),
        "null_reasons": reasons,
    }
    return FragmentInteractionResult(**(fields | changes))


def test_consistent_result_validates_and_returns_itself():
    result = _result()
    assert result.validate() is result
    assert result.schema_version == SCHEMA_VERSION == "fragments.result.v1"


@pytest.mark.parametrize("layer", ["hf", "mp2"])
def test_stored_interaction_energy_must_match_subsystems(layer):
    # 1e-10 Ha is above the 1e-12 Ha tolerance and below a loosened 1e-9 Ha one.
    result = _result()
    delta_e = dict(result.delta_e) | {layer: result.delta_e[layer] + 1e-10}
    with pytest.raises(ValueError, match=f"delta_e\\['{layer}'\\]"):
        replace(result, delta_e=delta_e).validate()


def test_interaction_energy_must_be_null_when_a_subsystem_energy_is_null():
    result = _result()
    monomer = replace(
        result.subsystems["A"],
        e_mp2=None,
        null_reasons=dict(result.subsystems["A"].null_reasons) | {"e_mp2": "not computed"},
    )
    subsystems = dict(result.subsystems) | {"A": monomer}
    with pytest.raises(ValueError, match="delta_e\\['mp2'\\]"):
        replace(result, subsystems=subsystems).validate()


@pytest.mark.parametrize("key", ["e_ccsd_t", "sqd", "active_space", "resources.qpe"])
def test_null_subsystem_field_needs_a_reason(key):
    result = _result()
    dimer = result.subsystems["AB"]
    reasons = {k: v for k, v in dimer.null_reasons.items() if k != key}
    subsystems = dict(result.subsystems) | {"AB": replace(dimer, null_reasons=reasons)}
    with pytest.raises(ValueError, match=f"AB.*{key}"):
        replace(result, subsystems=subsystems).validate()


@pytest.mark.parametrize("key", ["delta_e.casci", "errors.model", "correspondence"])
def test_null_result_field_needs_a_reason(key):
    result = _result()
    reasons = {k: v for k, v in result.null_reasons.items() if k != key}
    with pytest.raises(ValueError, match=key):
        replace(result, null_reasons=reasons).validate()


def test_classical_only_result_cannot_hold_an_sqd_record():
    result = _result()
    # Any non-null record breaks the rule; the check runs before the record's own checks.
    dimer = replace(result.subsystems["AB"], sqd=object())
    subsystems = dict(result.subsystems) | {"AB": dimer}
    with pytest.raises(ValueError, match="classical_only"):
        replace(result, subsystems=subsystems).validate()


def test_dimer_result_needs_exactly_the_three_counterpoise_subsystems():
    result = _result()
    subsystems = {k: v for k, v in result.subsystems.items() if k != "B"}
    with pytest.raises(ValueError, match="AB, A, B"):
        replace(result, subsystems=subsystems).validate()
