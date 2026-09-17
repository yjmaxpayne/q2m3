"""HF provenance labels must preserve values and the default QPE dispatch."""

from types import SimpleNamespace

import numpy as np
import pytest

from q2m3.core import QPEEngine, QuantumQMMM


@pytest.mark.parametrize("iterations, expected", [(2, 2), (8, 5)])
def test_hf_returns_label_both_result_and_simulated_convergence(iterations, expected):
    engine = QPEEngine(n_qubits=2, n_iterations=iterations)
    density = np.array([[1.0, 0.25], [0.25, 1.0]])
    result = engine.estimate_ground_state_energy(
        {"energy_hf": -1.234, "scf_result": SimpleNamespace(make_rdm1=lambda: density)}
    )
    assert result["method"] == "hf_reference"
    assert result["energy"] == -1.234
    assert result["density_matrix"] is density
    assert result["convergence"] == {
        "method": "hf_reference",
        "converged": True,
        "iterations": expected,
        "error_estimate": 0.001,
    }
    assert engine._simulate_qpe_convergence() == result["convergence"]


def test_default_workflow_still_dispatches_to_real_qpe(monkeypatch):
    workflow = QuantumQMMM(qm_atoms=[], mm_waters=0, qpe_config={"system_qubits": 2})
    density = np.eye(2)
    data = {"pennylane_hamiltonian": object(), "mol": object()}
    monkeypatch.setattr(workflow, "_build_qmmm_hamiltonian", lambda: data)
    monkeypatch.setattr(workflow, "_mulliken_analysis", lambda dm, mol: {})

    def real_qpe(received):
        assert received is data
        return {"energy": -1.5, "density_matrix": density, "convergence": {"converged": True}}

    def unexpected_hf(*args):
        pytest.fail("Default workflow was redirected to HF")

    monkeypatch.setattr(workflow, "_run_real_qpe", real_qpe)
    monkeypatch.setattr(workflow.qpe_engine, "estimate_ground_state_energy", unexpected_hf)
    result = workflow.compute_ground_state()
    assert workflow.qpe_config["use_real_qpe"] is True
    assert result["energy"] == -1.5
    assert result["density_matrix"] is density
