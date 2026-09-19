"""Pure SQD configuration, structural validation, and ownership contracts."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from q2m3.molecule import MoleculeConfig
from q2m3.sqd.config import (
    CCSDSeed,
    IntegralContext,
    LUCJConfig,
    ReferenceConfig,
    SQDConfig,
    _freeze_snapshot,
    validate_integral_inputs,
    validate_molecular_inputs,
)


def molecule(**updates):
    values = dict(
        name="H2",
        symbols=["H", "H"],
        coords=[[0, 0, 0], [0, 0, 0.7]],
        charge=0,
        active_electrons=2,
        active_orbitals=2,
    )
    return MoleculeConfig(**(values | updates))


def context(**updates):
    values = dict(
        frame_id="rhf",
        hamiltonian_id="a" * 64,
        active_indices=(0, 1),
        n_core_orbitals=0,
        vacuum_core_constant=0.5,
        delta_core_constant=0.0,
        embedding_mode="vacuum",
        fixed_mo=True,
        two_electron_tensor_fixed=True,
        hf_reference_kind="canonical_rhf",
        source={"nested": np.array([1.0, 2.0])},
    )
    return IntegralContext(**(values | updates))


def test_defaults_validation_and_frozen_configs():
    for config in (LUCJConfig(), ReferenceConfig(), SQDConfig(molecule())):
        assert config.validate() is config
        with pytest.raises(FrozenInstanceError):
            config.new_attribute = 1
    assert ReferenceConfig().sci_cutoffs == (1e-4, 1e-5)


@pytest.mark.parametrize(
    "field,value", [("shots", 0), ("shots", True), ("n_reps", -1), ("n_reps", 1.2)]
)
def test_lucj_rejects_invalid_counts(field, value):
    with pytest.raises(ValueError):
        LUCJConfig(**{field: value}).validate()


def test_pairs_are_copied_and_validated_against_orbitals():
    pairs = [[[0, 0], [0, 1]], []]
    config = LUCJConfig(interaction_pairs=pairs)
    pairs[0][0][0] = 99
    assert config.interaction_pairs == (((0, 0), (0, 1)), ())
    SQDConfig(molecule(), lucj=config).validate()
    with pytest.raises(ValueError):
        SQDConfig(molecule(), lucj=LUCJConfig(interaction_pairs=(((0, 2),), None))).validate()


@pytest.mark.parametrize(
    "pairs",
    [([],), (None, None, None), (((1, 0),), None), (((0, 1), (0, 1)), None), (((True, 1),), None)],
)
def test_invalid_pair_structure(pairs):
    with pytest.raises(ValueError):
        LUCJConfig(interaction_pairs=pairs).validate()


@pytest.mark.parametrize(
    "updates",
    [
        dict(active_electrons=1),
        dict(active_electrons=6),
        dict(active_electrons=None),
        dict(active_orbitals=0),
        dict(active_orbitals=True),
        dict(coords=[[0, 0, 0], [0, np.inf, 0]]),
        dict(symbols=[]),
        dict(charge=True),
    ],
)
def test_molecular_domain(updates):
    with pytest.raises(ValueError):
        SQDConfig(molecule(**updates)).validate()


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(seed=True),
        dict(seed=-1),
        dict(mode="auto"),
        dict(embedding_mode="vacuum"),
        dict(allow_large=1),
    ],
)
def test_run_configuration_domain(kwargs):
    with pytest.raises(ValueError):
        SQDConfig(molecule(), **kwargs).validate()


@pytest.mark.parametrize(
    "updates",
    [
        dict(wall_budget_s=0),
        dict(rss_budget_mb=np.inf),
        dict(sci_cutoffs=(1e-5, 1e-4)),
        dict(sci_cutoffs=(1, 0.1)),
        dict(allowed_tiers=("T3",)),
        dict(allowed_tiers=("T2",)),
        dict(allowed_tiers=("T0", "T0")),
        dict(plugins=("unknown",)),
        dict(plugins=("shci_dice",) * 2),
    ],
)
def test_reference_domain(updates):
    with pytest.raises(ValueError):
        ReferenceConfig(**updates).validate()


def test_snapshot_copies_molecule_and_nested_arrays():
    mol = molecule()
    values = {"molecule": mol, "nested": {"array": np.array([2.0])}}
    frozen = _freeze_snapshot(values)
    mol.coords[0][0] = 5
    values["nested"]["array"][0] = 3
    assert frozen["molecule"]["coords"][0][0] == 0
    assert frozen["nested"]["array"] == (2.0,)
    with pytest.raises(TypeError):
        frozen["nested"]["new"] = 1


@pytest.mark.parametrize("value", [np.nan, np.inf, {1: "bad"}, object(), 1j])
def test_snapshot_rejects_non_json_values(value):
    with pytest.raises(ValueError):
        _freeze_snapshot(value)


def test_mm_validation():
    validate_molecular_inputs(
        molecule(),
        mm_charges=np.array([1.0]),
        mm_coords=np.zeros((1, 3)),
        embedding_mode="full_oneelectron",
    )
    for kwargs in (
        dict(mm_charges=[1.0]),
        dict(mm_charges=[1.0], mm_coords=[[0, 0]]),
        dict(mm_charges=[np.nan], mm_coords=[[0, 0, 0]]),
    ):
        with pytest.raises(ValueError):
            validate_molecular_inputs(molecule(), **kwargs)


def test_integral_context_and_seed_own_their_inputs():
    source = {"a": [1]}
    ctx = context(source=source)
    source["a"][0] = 2
    assert ctx.source["a"] == (1,)
    t1, t2 = np.zeros((1, 1)), np.zeros((1, 1, 1, 1))
    seed = CCSDSeed(t1, t2, -1.0, -1.1, True, 0.0, "rhf", "a" * 64, "rccsd", "1")
    t1[0, 0] = 1
    assert seed.t1[0, 0] == 0
    with pytest.raises(ValueError):
        seed.t1[0, 0] = 2
    with pytest.raises(ValueError):
        seed.t1.setflags(write=True)


def test_integrals_structural_acceptance_and_failure_boundaries():
    kwargs = dict(norb=2, nelec=(1, 1), context=context(), seed_data=None, mode="reference_only")
    h1, h2 = np.eye(2), np.zeros((2,) * 4)
    validate_integral_inputs(h1, h2, 0.5, **kwargs)
    for change in (
        dict(nelec=(1, 0)),
        dict(norb=True),
        dict(nelec=(3, 3)),
        dict(mode="full"),
        dict(context=context(active_indices=(0,))),
        dict(context=context(vacuum_core_constant=0.6)),
    ):
        with pytest.raises(ValueError):
            validate_integral_inputs(h1, h2, 0.5, **(kwargs | change))
    for bad_h1, bad_h2 in (
        (h1.astype(complex), h2),
        (h1, np.zeros((2, 2))),
        (np.array([[1.0, 1.0], [0.0, 1.0]]), h2),
        (h1 * np.nan, h2),
    ):
        with pytest.raises(ValueError):
            validate_integral_inputs(bad_h1, bad_h2, 0.5, **kwargs)
    h2[0, 0, 0, 1] = 1
    with pytest.raises(ValueError):
        validate_integral_inputs(h1, h2, 0.5, **kwargs)


def test_full_embedding_config_can_be_validated_before_mm_inputs_arrive():
    config = SQDConfig(molecule(), embedding_mode="full_oneelectron")
    assert config.validate() is config
    with pytest.raises(ValueError):
        validate_molecular_inputs(config.molecule, embedding_mode=config.embedding_mode)


def test_config_snapshot_is_stable_after_input_molecule_mutation():
    mol = molecule()
    config = SQDConfig(mol)
    snapshot = config.snapshot()
    mol.coords[0][0] = 9
    assert snapshot["molecule"]["coords"][0][0] == 0
    assert config.molecule is not mol
    assert config.molecule.coords[0][0] == 0


@pytest.mark.parametrize(
    "change",
    [
        dict(frame_id=""),
        dict(hamiltonian_id="label"),
        dict(active_indices=(0, 0)),
        dict(n_core_orbitals=-1),
        dict(embedding_mode="bad"),
        dict(vacuum_core_constant=np.inf),
        dict(fixed_mo=1),
        dict(hf_reference_kind="rhf"),
    ],
)
def test_context_metadata_rejects_invalid_values(change):
    with pytest.raises(ValueError):
        context(**change)


def test_low_level_seed_shape_provenance_and_convergence_guards():
    from q2m3.sqd.exceptions import CCSDConvergenceError, ProvenanceMismatchError

    args = (np.eye(2), np.zeros((2,) * 4), 0.5)
    kwargs = dict(norb=2, nelec=(1, 1), context=context())

    def seed(**updates):
        values = dict(
            t1=np.zeros((1, 1)),
            t2=np.zeros((1, 1, 1, 1)),
            hf_energy=-1.0,
            ccsd_energy=-1.1,
            converged=True,
            residual_max_abs_ha=0.0,
            frame_id="rhf",
            hamiltonian_id="a" * 64,
            solver="rccsd",
            solver_version="1",
        )
        return CCSDSeed(**(values | updates))

    validate_integral_inputs(*args, **kwargs, seed_data=seed())
    with pytest.raises(ProvenanceMismatchError):
        validate_integral_inputs(*args, **kwargs, seed_data=seed(frame_id="other"))
    with pytest.raises(CCSDConvergenceError):
        validate_integral_inputs(*args, **kwargs, seed_data=seed(converged=False))
    validate_integral_inputs(
        *args, **kwargs, seed_data=seed(converged=False), mode="reference_only"
    )
    for updates in (
        dict(t1=np.zeros((2, 0)), t2=np.zeros((2, 2, 0, 0))),
        dict(t1=np.zeros((1, 1), dtype=complex)),
        dict(ccsd_energy=np.nan),
        dict(residual_max_abs_ha=-1.0),
    ):
        with pytest.raises(ValueError):
            validate_integral_inputs(*args, **kwargs, seed_data=seed(**updates))


@pytest.mark.parametrize(
    "updates",
    [dict(active_electrons=4, active_orbitals=4), dict(charge=1), dict(symbols=["Unknown", "H"])],
)
def test_molecular_electron_capacity_and_closed_shell(updates):
    with pytest.raises(ValueError):
        SQDConfig(molecule(**updates)).validate()


def test_package_exports_data_contracts():
    import q2m3.sqd as sqd

    assert sqd.SQDConfig is SQDConfig
    assert sqd.SQDResult.__name__ == "SQDResult"
    assert sqd.CCSDSeed is CCSDSeed
    assert {"SQDConfig", "SQDResult", "CCSDSeed"} <= set(sqd.__all__)


@pytest.mark.parametrize(
    "updates",
    [
        dict(source=[1]),
        dict(embedding_mode="full_oneelectron", hf_reference_kind="canonical_rhf"),
        dict(embedding_mode="diagonal", hf_reference_kind="canonical_rhf"),
    ],
)
def test_context_requires_mapping_source_and_honest_mm_reference_kind(updates):
    with pytest.raises(ValueError):
        context(**updates)


def test_full_mode_rejects_claimed_convergence_with_excessive_declared_residual():
    seed = CCSDSeed(
        np.zeros((1, 1)),
        np.zeros((1, 1, 1, 1)),
        -1.0,
        -1.1,
        True,
        1.1e-7,
        "rhf",
        "a" * 64,
        "rccsd",
        "1",
    )
    from q2m3.sqd.exceptions import CCSDConvergenceError

    with pytest.raises(CCSDConvergenceError):
        validate_integral_inputs(
            np.eye(2),
            np.zeros((2,) * 4),
            0.5,
            norb=2,
            nelec=(1, 1),
            context=context(),
            seed_data=seed,
        )


def test_pure_type_import_and_use_with_optional_backend_imports_blocked():
    import os
    import subprocess
    import sys
    import textwrap
    from pathlib import Path

    code = textwrap.dedent(
        """
        import importlib.abc
        import sys
        import q2m3  # Existing root-package imports are outside the SQD boundary.
        before = set(sys.modules)
        forbidden = {"ffsim", "qiskit", "qiskit_addon_sqd", "pennylane", "catalyst"}

        class BlockBackends(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".")[0] in forbidden:
                    raise ModuleNotFoundError("blocked backend: " + fullname, name=fullname)

        sys.meta_path.insert(0, BlockBackends())
        from q2m3.molecule import MoleculeConfig
        import q2m3.sqd as sqd
        config = sqd.SQDConfig(MoleculeConfig(
            "H2", ["H", "H"], [[0, 0, 0], [0, 0, .7]], 0, 2, 2))
        assert config.validate() is config
        assert config.snapshot()["molecule"]["active_electrons"] == 2
        assert all(hasattr(sqd, name) for name in sqd.__all__)
        assert "run_sqd" not in sqd.__all__
        assert "run_sqd_from_integrals" not in sqd.__all__
        added = set(sys.modules) - before
        assert not {name for name in added if name.split(".")[0] in forbidden}
    """
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "src")
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr
