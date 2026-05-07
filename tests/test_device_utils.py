# Copyright (c) 2026 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT

"""
Tests for device selection utilities.

TDD tests for the shared device_utils module that provides
unified device selection logic for QPE and RDM modules.
"""

import warnings

import pennylane as qml
import pytest


class TestSelectDevice:
    """Test select_device() function."""

    def test_default_qubit_always_works(self):
        """default.qubit should always be available."""
        from q2m3.core.device_utils import select_device

        dev = select_device("default.qubit", n_wires=4)
        assert dev is not None
        # PennyLane new API uses wires property
        assert len(dev.wires) == 4

    def test_default_qubit_with_set_shots(self):
        """default.qubit with shots set via qml.set_shots (PennyLane v0.43+ API)."""
        from q2m3.core.device_utils import select_device

        dev = select_device("default.qubit", n_wires=4)
        assert dev is not None

        # PennyLane v0.43+: shots are set via qml.set_shots() on QNode
        @qml.set_shots(100)
        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(0)
            return qml.sample(qml.PauliZ(0))

        result = circuit()
        assert result is not None
        assert len(result) == 100  # Should return 100 samples

    def test_select_device_accepts_seed(self):
        """select_device forwards a reproducibility seed to the PennyLane device."""
        from q2m3.core.device_utils import select_device

        dev = select_device("default.qubit", n_wires=2, seed=42)

        @qml.set_shots(20)
        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(0)
            return qml.sample(qml.PauliZ(0))

        result_a = circuit()
        result_b = circuit()
        assert result_a is not None
        assert result_b is not None

    def test_auto_selects_available_device(self):
        """auto mode should select an available device."""
        from q2m3.core.device_utils import select_device

        dev = select_device("auto", n_wires=4)
        assert dev is not None
        # Should select something (GPU > lightning.qubit > default.qubit)

    def test_auto_returns_consistent_device(self):
        """auto mode should return consistent device type."""
        from q2m3.core.device_utils import (
            HAS_LIGHTNING_GPU,
            HAS_LIGHTNING_QUBIT,
            select_device,
        )

        dev = select_device("auto", n_wires=4)
        dev_name = dev.name

        # Verify it matches expected hierarchy
        if HAS_LIGHTNING_GPU:
            assert "gpu" in dev_name.lower() or "lightning" in dev_name.lower()
        elif HAS_LIGHTNING_QUBIT:
            assert "lightning" in dev_name.lower() or "qubit" in dev_name.lower()
        else:
            assert "qubit" in dev_name.lower()

    def test_lightning_qubit_fallback_warning(self):
        """Warning when lightning.qubit unavailable."""
        from q2m3.core.device_utils import HAS_LIGHTNING_QUBIT, select_device

        if HAS_LIGHTNING_QUBIT:
            pytest.skip("lightning.qubit is available, cannot test fallback")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            dev = select_device("lightning.qubit", n_wires=4)
            assert len(w) == 1
            assert "not available" in str(w[0].message).lower()
            assert dev is not None  # Should still return a fallback device

    def test_catalyst_compatible_fallback(self):
        """With use_catalyst=True, fallback should prefer lightning.qubit."""
        from q2m3.core.device_utils import (
            HAS_LIGHTNING_GPU,
            HAS_LIGHTNING_QUBIT,
            select_device,
        )

        if HAS_LIGHTNING_GPU:
            pytest.skip("GPU available, no fallback needed")

        # When GPU unavailable but Catalyst requested, should use lightning.qubit
        # or default.qubit if lightning.qubit also unavailable
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            dev = select_device("lightning.gpu", n_wires=4, use_catalyst=True)
            assert dev is not None
            # If lightning.qubit available, should fallback to it (not default.qubit)
            if HAS_LIGHTNING_QUBIT:
                assert "lightning" in dev.name.lower()


class TestDeviceAvailability:
    """Test device availability detection."""

    def test_has_lightning_gpu_is_bool(self):
        """HAS_LIGHTNING_GPU should be a boolean."""
        from q2m3.core.device_utils import HAS_LIGHTNING_GPU

        assert isinstance(HAS_LIGHTNING_GPU, bool)

    def test_has_lightning_qubit_is_bool(self):
        """HAS_LIGHTNING_QUBIT should be a boolean."""
        from q2m3.core.device_utils import HAS_LIGHTNING_QUBIT

        assert isinstance(HAS_LIGHTNING_QUBIT, bool)

    def test_get_best_available_device(self):
        """get_best_available_device() returns a valid device name."""
        from q2m3.core.device_utils import get_best_available_device

        device_name = get_best_available_device()
        assert device_name in ["lightning.gpu", "lightning.qubit", "default.qubit"]


@pytest.mark.gpu
class TestGPUDevice:
    """GPU-specific device tests."""

    def test_lightning_gpu_creation(self):
        """Test lightning.gpu device can be created."""
        from q2m3.core.device_utils import HAS_LIGHTNING_GPU, select_device

        if not HAS_LIGHTNING_GPU:
            pytest.skip("lightning.gpu not available")

        dev = select_device("lightning.gpu", n_wires=4)
        assert dev is not None
        assert "gpu" in dev.name.lower() or "lightning" in dev.name.lower()

    def test_auto_selects_gpu_when_available(self):
        """auto mode should select GPU when available."""
        from q2m3.core.device_utils import HAS_LIGHTNING_GPU, select_device

        if not HAS_LIGHTNING_GPU:
            pytest.skip("lightning.gpu not available")

        dev = select_device("auto", n_wires=4)
        # When GPU available, auto should select it
        assert "gpu" in dev.name.lower() or "lightning" in dev.name.lower()


class TestSelectDeviceIntegration:
    """Integration tests for device selection with actual circuits."""

    def test_device_works_with_qnode(self):
        """Selected device should work with qml.qnode."""
        from q2m3.core.device_utils import select_device

        dev = select_device("auto", n_wires=2)

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        result = circuit()
        assert result is not None

    def test_device_with_trotter_product(self):
        """Device should support TrotterProduct for RDM measurement."""
        from q2m3.core.device_utils import select_device

        dev = select_device("auto", n_wires=2)

        # Simple test Hamiltonian
        H = qml.Hamiltonian([1.0, 0.5], [qml.PauliZ(0), qml.PauliX(1)])

        @qml.qnode(dev)
        def circuit():
            qml.BasisState([1, 0], wires=[0, 1])
            qml.TrotterProduct(H, time=0.1, n=2, order=2)
            return qml.expval(qml.PauliZ(0))

        result = circuit()
        assert result is not None


class TestDeviceFallbackPaths:
    """Test device fallback warning paths."""

    def test_unrecognized_device_type_fallback(self):
        """Unrecognized device_type should fallback to default.qubit."""
        from q2m3.core.device_utils import select_device

        dev = select_device("unknown_device", n_wires=4)
        assert dev is not None
        assert "qubit" in dev.name.lower()

    def test_catalyst_fallback_without_lightning(self):
        """Test Catalyst fallback when lightning.qubit unavailable."""
        from q2m3.core.device_utils import HAS_LIGHTNING_QUBIT, select_device

        # This path is hard to test deterministically since it depends on installed packages
        # We can only test the expected behavior when lightning is NOT available
        if HAS_LIGHTNING_QUBIT:
            pytest.skip("lightning.qubit available, cannot test unavailable path")

        # When Catalyst requested but lightning unavailable, should still return valid device
        dev = select_device("auto", n_wires=4, use_catalyst=True)
        assert dev is not None
