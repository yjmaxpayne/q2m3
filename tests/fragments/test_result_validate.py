"""SQD-layer rules of the fragment result records.

The SQD results are built directly, so these checks run no SQD. Each subsystem holds the
same synthetic SQD energies; only the Hamiltonian digests tell the subsystems apart.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from q2m3.fragments.result import (
    ENERGY_LAYERS,
    ActiveSpaceRecord,
    ErrorDecomposition,
    ExternalReference,
    FragmentInteractionResult,
    SubsystemResult,
    interaction_energy,
)
from q2m3.sqd.result import SQDResult
from tests.sqd.test_result import result_data

HAMILTONIAN_IDS = {"AB": "a" * 64, "A": "b" * 64, "B": "c" * 64}
NOT_COMPUTED = "not computed"
# Provenance keys every result needs; a full-mode result also needs ``ccsd_t_cost``.
PROVENANCE_KEYS = (
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
FULL_PROVENANCE = {key: "recorded" for key in (*PROVENANCE_KEYS, "ccsd_t_cost")}


def sqd_record(name: str, **changes) -> SQDResult:
    context = {"hamiltonian_id": HAMILTONIAN_IDS[name]}
    return SQDResult(**(result_data() | {"provenance": {"context": context}} | changes))


def full_subsystem(name: str, sqd: SQDResult, **changes) -> SubsystemResult:
    record = ActiveSpaceRecord(
        nelecas=2,
        ncas=2,
        ncore=0,
        ao_labels=(),
        minao_indices=(),
        hamiltonian_id=sqd.provenance["context"]["hamiltonian_id"],
    )
    fields = {
        "name": name,
        "real_atoms": (0,),
        "ghost_atoms": (),
        "n_ao": 2,
        "e_hf": sqd.hf_energy,
        "e_mp2": sqd.hf_energy - 0.01,
        "e_ccsd_t": None,
        "e_casci": sqd.baseline_energy,
        "e_sqd": sqd.sqd_energy,
        "active_space": record,
        "sqd": sqd,
        "resources": {},
        "null_reasons": {
            "e_ccsd_t": NOT_COMPUTED,
            "resources.qpe": NOT_COMPUTED,
            "resources.sqd": NOT_COMPUTED,
        },
    }
    return SubsystemResult(**(fields | changes))


def full_result(subsystems=None, **changes) -> FragmentInteractionResult:
    if subsystems is None:
        subsystems = {name: full_subsystem(name, sqd_record(name)) for name in HAMILTONIAN_IDS}
    delta_e = {layer: interaction_energy(subsystems, layer) for layer in ENERGY_LAYERS}
    reasons = {f"delta_e.{layer}": NOT_COMPUTED for layer, v in delta_e.items() if v is None}
    algorithm = None
    if delta_e["sqd"] is not None and delta_e["casci"] is not None:
        algorithm = delta_e["sqd"] - delta_e["casci"]
    else:
        reasons["errors.algorithm"] = NOT_COMPUTED
    reasons |= {f"errors.{term}": NOT_COMPUTED for term in ("active_space", "basis", "model")}
    fields = {
        "fragment_id": "water-dimer",
        "mode": "full",
        "basis": "cc-pvdz",
        "subsystems": subsystems,
        "delta_e": delta_e,
        "references": (),
        "errors": ErrorDecomposition(algorithm=algorithm),
        "correspondence": None,
        "provenance": FULL_PROVENANCE,
        "null_reasons": reasons | {"correspondence": NOT_COMPUTED},
    }
    return FragmentInteractionResult(**(fields | changes))


def _with_subsystem(name: str, subsystem: SubsystemResult) -> FragmentInteractionResult:
    subsystems = {key: full_subsystem(key, sqd_record(key)) for key in HAMILTONIAN_IDS}
    return full_result(subsystems | {name: subsystem})


def test_consistent_full_result_validates_and_returns_itself():
    result = full_result()
    assert result.errors.algorithm is not None
    assert result.validate() is result


def test_sqd_record_is_validated_with_its_own_rules():
    # A completed SQD result needs its baseline energy; SQDResult.validate() enforces it.
    sqd = sqd_record("A", baseline_energy=None)
    subsystem = full_subsystem("A", sqd, e_casci=None)
    subsystem = replace(subsystem, null_reasons=dict(subsystem.null_reasons) | {"e_casci": "x"})
    with pytest.raises(ValueError, match="baseline_energy"):
        subsystem.validate()


def test_sqd_record_must_be_exactly_an_sqd_result():
    class RelabeledResult(SQDResult):
        pass

    sqd = RelabeledResult(**vars(sqd_record("A")))
    with pytest.raises(ValueError, match="RelabeledResult"):
        full_subsystem("A", sqd).validate()


def test_stored_sqd_interaction_energy_must_match_the_subsystems():
    result = full_result()
    shifted = dict(result.delta_e) | {"sqd": result.delta_e["sqd"] + 1e-6}
    with pytest.raises(ValueError, match=r"delta_e\['sqd'\]"):
        replace(result, delta_e=shifted).validate()


def test_active_space_must_be_the_one_the_sqd_run_used():
    subsystem = full_subsystem("A", sqd_record("A"))
    other_space = replace(subsystem.active_space, hamiltonian_id=HAMILTONIAN_IDS["B"])
    with pytest.raises(ValueError, match="hamiltonian_id"):
        replace(subsystem, active_space=other_space).validate()


def test_e_sqd_must_be_the_sqd_energy_of_the_record():
    sqd = sqd_record("A")
    assert sqd.baseline_energy != sqd.sqd_energy
    with pytest.raises(ValueError, match="e_sqd"):
        full_subsystem("A", sqd, e_sqd=sqd.baseline_energy).validate()


def test_e_casci_must_be_the_t0_baseline_of_the_record():
    sqd = sqd_record("A")
    with pytest.raises(ValueError, match="e_casci"):
        full_subsystem("A", sqd, e_casci=sqd.baseline_energy + 1e-3).validate()


def test_stored_algorithm_error_must_match_the_interaction_energies():
    result = full_result()
    shifted = ErrorDecomposition(algorithm=result.errors.algorithm + 1e-6)
    with pytest.raises(ValueError, match="errors.algorithm"):
        replace(result, errors=shifted).validate()


def test_full_result_needs_an_sqd_record_for_every_subsystem():
    subsystem = full_subsystem("AB", sqd_record("AB"))
    skipped = {key: "SQD stage skipped" for key in ("e_casci", "e_sqd", "sqd")}
    without_sqd = replace(
        subsystem,
        e_casci=None,
        e_sqd=None,
        sqd=None,
        null_reasons=dict(subsystem.null_reasons) | skipped,
    )
    with pytest.raises(ValueError, match="no SQD record for AB"):
        _with_subsystem("AB", without_sqd).validate()


SAME_BASIS = ExternalReference(
    fragment_id="water-dimer",
    method="CCSD(T)",
    basis="cc-pvdz",
    delta_e=-0.0049,
    category="same_basis_full_space",
)
EXTERNAL = replace(SAME_BASIS, basis="CBS", delta_e=-0.0050, category="external")


def referenced_result(*, references, source, same=None, external, ccsd_t=None, **shifts):
    """Full result whose error terms are formed by hand from ``same`` and ``external``.

    ``ccsd_t`` sets the PySCF CCSD(T) interaction energy; ``same=None`` then uses it as the
    same-basis value. ``shifts`` are added to the stored error terms.
    """
    subsystems = {name: full_subsystem(name, sqd_record(name)) for name in HAMILTONIAN_IDS}
    if ccsd_t is not None:
        energies = {"AB": -2.0 + ccsd_t, "A": -1.0, "B": -1.0}
        subsystems = {
            name: replace(sub, e_ccsd_t=energies[name]) for name, sub in subsystems.items()
        }
    result = full_result(subsystems, references=tuple(references))
    same = result.delta_e["ccsd_t"] if same is None else same
    casci = result.delta_e["casci"]
    terms = {"active_space": casci - same, "basis": same - external, "model": casci - external}
    errors = replace(result.errors, **{k: v + shifts.get(k, 0.0) for k, v in terms.items()})
    provenance = FULL_PROVENANCE | {"same_basis_reference_source": source}
    return replace(result, errors=errors, provenance=provenance)


def _both_imported(**shifts) -> FragmentInteractionResult:
    return referenced_result(
        references=(SAME_BASIS, EXTERNAL),
        source="imported",
        same=SAME_BASIS.delta_e,
        external=EXTERNAL.delta_e,
        **shifts,
    )


def test_error_terms_from_both_references_validate():
    result = _both_imported()
    assert result.validate() is result


def test_model_error_without_an_external_reference_must_stay_null():
    result = full_result()
    with pytest.raises(ValueError, match="errors.model"):
        replace(result, errors=replace(result.errors, model=0.0)).validate()


@pytest.mark.parametrize("term", ["active_space", "basis", "model"])
def test_stored_error_term_must_match_the_references(term):
    # The message of the recomputation check, not of the sum check, which also fails.
    with pytest.raises(ValueError, match=f"errors.{term} = .* does not match the recomputed"):
        _both_imported(**{term: 1e-11}).validate()


def test_model_error_must_equal_active_space_plus_basis_error():
    # Each term is within the 1e-12 Ha recomputation tolerance; their sum is not.
    result = _both_imported(active_space=0.9e-12, basis=0.9e-12, model=-0.9e-12)
    with pytest.raises(ValueError, match=r"errors.active_space \+ errors.basis"):
        result.validate()


def test_pyscf_ccsd_t_takes_precedence_over_an_imported_same_basis_value():
    references = (SAME_BASIS, EXTERNAL)
    computed = referenced_result(
        references=references, source="pyscf_ccsd_t", external=EXTERNAL.delta_e, ccsd_t=-0.0047
    )
    assert computed.validate() is computed
    imported = referenced_result(
        references=references,
        source="imported",
        same=SAME_BASIS.delta_e,
        external=EXTERNAL.delta_e,
        ccsd_t=-0.0047,
    )
    with pytest.raises(ValueError, match="errors.active_space"):
        imported.validate()


def test_recorded_same_basis_source_must_be_the_one_used():
    result = referenced_result(
        references=(EXTERNAL,), source="imported", external=EXTERNAL.delta_e, ccsd_t=-0.0047
    )
    with pytest.raises(ValueError, match="same_basis_reference_source"):
        result.validate()


@pytest.mark.parametrize("key", [*PROVENANCE_KEYS, "ccsd_t_cost"])
def test_full_result_provenance_needs_every_required_key(key):
    provenance = {k: v for k, v in FULL_PROVENANCE.items() if k != key}
    with pytest.raises(ValueError, match=key):
        full_result(provenance=provenance).validate()


def test_reference_category_must_be_known():
    unknown = replace(EXTERNAL, category="cbs")
    with pytest.raises(ValueError, match="category"):
        full_result(references=(unknown,)).validate()


def test_at_most_one_reference_per_category():
    second = replace(EXTERNAL, method="S22B", delta_e=-0.0051)
    with pytest.raises(ValueError, match="external"):
        full_result(references=(EXTERNAL, second)).validate()
