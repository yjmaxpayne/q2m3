import pennylane as qml
import pytest

from q2m3.core.hamiltonian_utils import build_operator_index_map, decompose_hamiltonian


def _make_test_hamiltonian():
    """Create a simple 2-qubit Hamiltonian: 0.5*I + 0.3*Z(0) + 0.2*Z(1)."""
    return qml.ops.op_math.Sum(
        qml.ops.SProd(0.5, qml.Identity(wires=[0, 1])),
        qml.ops.SProd(0.3, qml.PauliZ(wires=0)),
        qml.ops.SProd(0.2, qml.PauliZ(wires=1)),
    )


def test_decompose_hamiltonian_returns_coeffs_ops():
    H = _make_test_hamiltonian()
    coeffs, ops = decompose_hamiltonian(H)
    assert len(coeffs) == 3
    assert len(ops) == 3
    assert abs(coeffs[0] - 0.5) < 1e-8


def test_decompose_hamiltonian_coeffs_are_float():
    H = _make_test_hamiltonian()
    coeffs, ops = decompose_hamiltonian(H)
    assert all(isinstance(c, float) for c in coeffs)


def test_build_operator_index_map_basic():
    H = _make_test_hamiltonian()
    coeffs, ops = decompose_hamiltonian(H)
    n_system_qubits = 2
    index_map, ext_coeffs, ext_ops = build_operator_index_map(ops, n_system_qubits, coeffs)

    assert "identity_idx" in index_map
    assert "z_wire_idx" in index_map
    assert 0 in index_map["z_wire_idx"]
    assert 1 in index_map["z_wire_idx"]


def test_build_operator_index_map_fills_missing_z():
    """Missing Z terms should be appended with coeff=0.0."""
    # Only Z(0), missing Z(1)
    H = qml.ops.op_math.Sum(
        qml.ops.SProd(0.5, qml.Identity(wires=[0, 1])),
        qml.ops.SProd(0.3, qml.PauliZ(wires=0)),
    )
    coeffs, ops = decompose_hamiltonian(H)
    n_system_qubits = 2
    index_map, ext_coeffs, ext_ops = build_operator_index_map(ops, n_system_qubits, coeffs)

    assert 1 in index_map["z_wire_idx"]
    # The appended coeff should be 0.0
    z1_idx = index_map["z_wire_idx"][1]
    assert abs(ext_coeffs[z1_idx]) < 1e-8


def test_build_operator_index_map_no_identity_raises():
    """Missing Identity term should raise ValueError."""
    ops = [qml.PauliZ(wires=0)]
    coeffs = [1.0]
    with pytest.raises(ValueError, match="Identity"):
        build_operator_index_map(ops, 1, coeffs)


def test_hamiltonian_utils_re_export():
    """Verify re-export returns the same function objects."""
    from examples.mc_solvation.energy import build_operator_index_map as OldBuild
    from examples.mc_solvation.energy import decompose_hamiltonian as OldDecompose
    from q2m3.core.hamiltonian_utils import build_operator_index_map as NewBuild
    from q2m3.core.hamiltonian_utils import decompose_hamiltonian as NewDecompose

    assert OldDecompose is NewDecompose, "re-export must return same function"
    assert OldBuild is NewBuild, "re-export must return same function"
