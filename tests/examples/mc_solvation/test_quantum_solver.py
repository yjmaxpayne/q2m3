# Copyright (c) 2025 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT
"""
Unit tests for mc_solvation.quantum_solver module.

Tests verify the abstract solver interface, result container,
and factory function.
"""

import numpy as np
import pytest

from examples.mc_solvation.quantum_solver import (
    QuantumSolver,
    SolverResult,
    create_solver,
)


class TestSolverResult:
    """Tests for SolverResult dataclass."""

    def test_create_result(self):
        """Should create result with required fields."""
        result = SolverResult(
            energy=-1.117,
            converged=True,
            n_evaluations=100,
            method="QPE",
        )
        assert result.energy == -1.117
        assert result.converged is True
        assert result.n_evaluations == 100
        assert result.method == "QPE"
        assert result.metadata == {}

    def test_result_with_metadata(self):
        """Should accept metadata dictionary."""
        metadata = {"n_shots": 50, "samples": np.array([1, 0, 1])}
        result = SolverResult(
            energy=-75.3,
            converged=True,
            n_evaluations=1,
            method="QPE",
            metadata=metadata,
        )
        assert result.metadata["n_shots"] == 50
        assert np.array_equal(result.metadata["samples"], [1, 0, 1])

    def test_result_unconverged(self):
        """Should represent unconverged result."""
        result = SolverResult(
            energy=-1.0,
            converged=False,
            n_evaluations=50,
            method="VQE",
        )
        assert result.converged is False


class TestQuantumSolverInterface:
    """Tests for QuantumSolver abstract base class."""

    def test_cannot_instantiate_abstract(self):
        """Should not be able to instantiate abstract class."""
        with pytest.raises(TypeError):
            QuantumSolver()

    def test_subclass_must_implement_solve(self):
        """Subclass without solve() should raise error."""

        class IncompleteSolver(QuantumSolver):
            @property
            def name(self) -> str:
                return "Incomplete"

            def get_compiled_circuit(self):
                return None

        with pytest.raises(TypeError):
            IncompleteSolver()

    def test_concrete_subclass_works(self):
        """Properly implemented subclass should work."""

        class MockSolver(QuantumSolver):
            @property
            def name(self) -> str:
                return "Mock"

            def solve(self, hamiltonian, hf_state, n_qubits, e_ref=0.0):
                return SolverResult(
                    energy=e_ref - 1.0,
                    converged=True,
                    n_evaluations=1,
                    method=self.name,
                )

            def get_compiled_circuit(self):
                return None

        solver = MockSolver()
        assert solver.name == "Mock"

        result = solver.solve(None, np.array([1, 1, 0, 0]), 4, e_ref=-75.0)
        assert result.energy == -76.0
        assert result.method == "Mock"


class TestCreateSolver:
    """Tests for create_solver factory function."""

    def test_create_qpe_solver(self):
        """Should create QPE solver."""
        solver = create_solver("qpe")
        assert solver.name == "QPE"

    def test_create_qpe_case_insensitive(self):
        """Method name should be case-insensitive."""
        solver1 = create_solver("QPE")
        solver2 = create_solver("qpe")
        solver3 = create_solver("Qpe")

        assert solver1.name == solver2.name == solver3.name == "QPE"

    def test_create_with_config(self):
        """Should accept configuration object."""
        from examples.mc_solvation.qpe_solver import QPESolverConfig

        config = QPESolverConfig(n_estimation_wires=6, n_shots=100)
        solver = create_solver("qpe", config)

        assert solver.config.n_estimation_wires == 6
        assert solver.config.n_shots == 100

    def test_unknown_method_raises(self):
        """Should raise ValueError for unknown method."""
        with pytest.raises(ValueError, match="Unknown solver method"):
            create_solver("unknown")

    def test_error_message_suggests_alternatives(self):
        """Error message should list available methods."""
        with pytest.raises(ValueError, match="qpe"):
            create_solver("invalid")
