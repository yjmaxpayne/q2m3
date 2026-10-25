"""Resource policy, model domain, and allocation-order checks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from q2m3.sqd import resources
from q2m3.sqd.exceptions import ResourceLimitError, ResourceModelDomainError


class ConstantModel:
    model_id = "policy-test"

    def __init__(self, value):
        self.value = value

    def upper_bound_mb(self, *args, **kwargs):
        if isinstance(self.value, bool):
            return self.value
        return self.value + kwargs["retained_mb"]


def _guarded_call(operation, norb, nelec, **kwargs):
    resources.guard_allocation(norb, nelec, **kwargs)
    return operation()


@pytest.mark.parametrize(
    "value,override,expected",
    [
        (2047.999, False, "pass"),
        (2048, False, "warn"),
        (8191.999, False, "warn"),
        (8192, False, "reject"),
        (8192, True, "warn"),
        (12287.999, True, "warn"),
        (12288, True, "reject"),
        (12288, False, "reject"),
    ],
)
def test_decimal_policy_equalities(value, override, expected):
    kwargs = dict(
        stage="prepare", model=ConstantModel(value), host_available_mb=20_000, allow_large=override
    )
    if expected == "reject":
        with pytest.raises(ResourceLimitError, match="predicted.*budget.*norb=4"):
            resources.guard_allocation(4, (2, 2), **kwargs)
    elif expected == "warn":
        with pytest.warns(RuntimeWarning, match="2048"):
            assert resources.guard_allocation(4, (2, 2), **kwargs) == value
    else:
        assert resources.guard_allocation(4, (2, 2), **kwargs) == value


@pytest.mark.parametrize("cap", ["host_available_mb", "rss_budget_mb"])
def test_override_cannot_cross_host_or_user_cap(cap):
    kwargs = dict(
        stage="prepare", model=ConstantModel(1000), host_available_mb=20_000, allow_large=True
    )
    kwargs[cap] = 1000
    with pytest.raises(ResourceLimitError):
        resources.guard_allocation(4, (2, 2), **kwargs)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, True])
def test_invalid_model_output_is_never_a_bound(value):
    with pytest.raises(ResourceModelDomainError):
        resources.estimate_rss_mb(4, (2, 2), stage="prepare", model=ConstantModel(value))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"stage": "missing"},
        {"stage": "diagonalize"},
        {"stage": "comparison"},
        {"stage": "reference"},
        {"stage": "prepare", "retained_mb": -1},
        {"stage": "diagonalize", "subspace_dims": (7, 2)},
        {"stage": "diagonalize", "subspace_dims": (True, 2)},
    ],
)
def test_missing_or_invalid_prediction_inputs_rejected(kwargs):
    with pytest.raises((ValueError, ResourceModelDomainError)):
        resources.estimate_rss_mb(4, (2, 2), model=ConstantModel(1), **kwargs)


def test_retained_arrays_are_added_once():
    assert (
        resources.estimate_rss_mb(
            4, (2, 2), stage="prepare", retained_mb=37, model=ConstantModel(100)
        )
        == 137
    )


def test_addon_prebound_covers_recovery_spin_and_unlimited_batches():
    # Six orbitals, two electrons per spin: all 15 choose-spin strings can
    # appear after recovery, even if the original batch contains one pair.
    assert resources.addon_subspace_bound(6, (2, 2)) == (15, 15)
    assert resources.addon_subspace_bound(6, (2, 2), max_dim=(7, 9)) == (7, 9)
    assert resources.addon_subspace_bound(6, (2, 2), max_dim=8) == (8, 8)
    with pytest.raises(ValueError):
        resources.addon_subspace_bound(6, (2, 2), max_dim=0)


def test_empirical_model_domain_and_retained():
    artifact = json.loads(
        Path(resources.__file__).with_name("resource_calibration.json").read_text()
    )
    model = resources.load_resource_model(versions=artifact["versions"])
    small = resources.estimate_rss_mb(
        10, (5, 5), stage="diagonalize", subspace_dims=(20, 60), model=model
    )
    big = resources.estimate_rss_mb(
        10, (5, 5), stage="diagonalize", subspace_dims=(252, 252), model=model
    )
    # Independent floor: 12 float64 Davidson vectors over the additional
    # 62,304 determinant pairs require 5,981,184 bytes.
    assert big - small > 5.981184
    retained = resources.estimate_rss_mb(
        10, (5, 5), stage="diagonalize", subspace_dims=(20, 60), retained_mb=80, model=model
    )
    assert retained - small == pytest.approx(80)
    rectangular = resources.estimate_rss_mb(
        10, (5, 5), stage="diagonalize", subspace_dims=(30, 40), model=model
    )
    assert small > rectangular  # same product, different link-table allocation
    with pytest.raises(ResourceModelDomainError):
        resources.guard_allocation(
            16, (8, 8), stage="prepare", model=model, host_available_mb=50_000
        )
    with pytest.raises(ResourceModelDomainError):
        resources.guard_allocation(
            16, (8, 8), stage="prepare", model=model, host_available_mb=50_000, allow_large=True
        )
    with pytest.raises(ResourceModelDomainError):
        resources.estimate_rss_mb(
            4, (2, 2), stage="reference", solver_method="unknown", model=model
        )


def test_runtime_profile_and_version_mismatch_rejected():
    # Pass the shipped manifest so this checks the profile, not the installed extras.
    artifact = json.loads(
        Path(resources.__file__).with_name("resource_calibration.json").read_text()
    )
    with pytest.raises(ResourceModelDomainError, match="shots"):
        resources.load_resource_model(versions=artifact["versions"], profile={"shots": 500_000})
    with pytest.raises(ResourceModelDomainError, match="dependency versions differ"):
        resources.load_resource_model(versions={"ffsim": "0.0.1"})


@pytest.mark.parametrize(
    "package,installed,accepted",
    [
        ("pennylane-catalyst", None, True),
        ("ffsim", None, False),
        ("pennylane-catalyst", "0.0.1", False),
        ("ffsim", "0.0.1", False),
    ],
)
def test_installed_profile_allows_only_absent_optional_catalyst(
    monkeypatch, package, installed, accepted
):
    artifact = json.loads(
        Path(resources.__file__).with_name("resource_calibration.json").read_text()
    )

    def installed_version(name):
        if name == package:
            if installed is None:
                raise resources.PackageNotFoundError(name)
            return installed
        return artifact["versions"][name]

    monkeypatch.setattr(resources, "version", installed_version)
    if accepted:
        assert resources.load_resource_model().model_id == artifact["model_id"]
    else:
        with pytest.raises(ResourceModelDomainError):
            resources.load_resource_model()


def test_explicit_manifest_still_requires_calibrated_catalyst_version():
    artifact = json.loads(
        Path(resources.__file__).with_name("resource_calibration.json").read_text()
    )
    explicit = dict(artifact["versions"])
    explicit.pop("pennylane-catalyst")
    with pytest.raises(ResourceModelDomainError, match="dependency versions differ"):
        resources.load_resource_model(versions=explicit)


@pytest.mark.sqd
def test_installed_environment_accepts_shipped_model():
    assert resources.load_resource_model().model_id


def test_domain_model_cannot_be_relabelled_as_validated(tmp_path):
    path = tmp_path / "unvalidated.json"
    path.write_text(json.dumps({"schema": "sqd.rss.v1", "status": "training_only"}))
    with pytest.raises(ResourceModelDomainError, match="unvalidated"):
        resources.load_resource_model(path)


def test_unavailable_prediction_reports_budget_and_dimensions():
    with pytest.raises(
        ResourceModelDomainError, match="predicted=unavailable, budget=1000.*subspace_dims=None"
    ):
        resources.guard_allocation(
            4, (2, 2), stage="diagonalize", model=ConstantModel(1), host_available_mb=1000
        )


@pytest.mark.parametrize("invalid", [3.5, None])
def test_invalid_dimension_container_fails_with_domain_error(invalid):
    with pytest.raises(ResourceModelDomainError):
        resources.estimate_rss_mb(
            4, (2, 2), stage="diagonalize", subspace_dims=invalid, model=ConstantModel(1)
        )
    if invalid is not None:
        with pytest.raises(ValueError):
            resources.addon_subspace_bound(4, (2, 2), max_dim=invalid)


@pytest.mark.sqd
def test_spin_symmetrization_and_recovery_stay_within_preallocation_bound():
    import numpy as np
    from qiskit.primitives import BitArray
    from qiskit_addon_sqd.fermion import diagonalize_fermionic_hamiltonian, solve_sci

    model = resources.load_resource_model()
    bits = np.array([[0, 0, 0, 1, 0, 0, 1, 0], [1, 1, 0, 0, 0, 0, 0, 1]] * 10, dtype=bool)
    cap = resources.addon_subspace_bound(4, (1, 1), max_dim=3)
    checked = []

    def solver(batches, h1, h2, norb, nelec):
        outputs = []
        for a, b in batches:
            dims = (len(a), len(b))
            assert dims[0] <= cap[0] and dims[1] <= cap[1]
            checked.append(dims)
            outputs.append(
                _guarded_call(
                    lambda a=a, b=b: solve_sci((a, b), h1, h2, norb, nelec, max_space=12),
                    norb,
                    nelec,
                    stage="diagonalize",
                    subspace_dims=dims,
                    model=model,
                    host_available_mb=20_000,
                )
            )
        return outputs

    output = _guarded_call(
        lambda: diagonalize_fermionic_hamiltonian(
            np.diag([0.0, 1.0, 2.0, 3.0]),
            np.zeros((4,) * 4),
            BitArray.from_bool_array(bits, order="big"),
            8,
            4,
            (1, 1),
            max_dim=3,
            num_batches=2,
            max_iterations=2,
            symmetrize_spin=True,
            carryover_threshold=0,
            include_configurations=[1],
            sci_solver=solver,
            seed=42,
        ),
        4,
        (1, 1),
        stage="diagonalize",
        subspace_dims=cap,
        model=model,
        host_available_mb=20_000,
    )
    assert checked and all(a == b for a, b in checked)
    assert np.isfinite(output.energy)
