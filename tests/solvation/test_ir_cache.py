# Copyright (c) 2026 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT

"""Tests for q2m3.solvation.ir_cache module."""

import pytest

from q2m3.solvation.config import MoleculeConfig, QPEConfig, SolvationConfig
from q2m3.solvation.ir_cache import (
    _serialize_config_for_subprocess,
    cache_path_for_config,
    compute_cache_key,
    default_cache_dir,
    is_cache_available,
)


@pytest.fixture
def h2_config():
    """Minimal H2 SolvationConfig for cache tests."""
    return SolvationConfig(
        molecule=MoleculeConfig(
            name="H2",
            symbols=["H", "H"],
            coords=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
            charge=0,
        ),
        qpe_config=QPEConfig(n_estimation_wires=3, n_trotter_steps=2),
        hamiltonian_mode="fixed",
        n_waters=3,
        n_mc_steps=5,
        verbose=False,
    )


@pytest.fixture
def h3o_config():
    """H3O+ config with different active space."""
    return SolvationConfig(
        molecule=MoleculeConfig(
            name="H3O+",
            symbols=["O", "H", "H", "H"],
            coords=[
                [0.0, 0.0, 0.0],
                [0.0, 0.757, 0.587],
                [0.0, -0.757, 0.587],
                [0.0, 0.0, -0.371],
            ],
            charge=1,
            active_electrons=4,
            active_orbitals=4,
        ),
        qpe_config=QPEConfig(n_estimation_wires=4, n_trotter_steps=3),
        hamiltonian_mode="dynamic",
        verbose=False,
    )


# ==========================================================================
# TestDefaultCacheDir
# ==========================================================================


class TestDefaultCacheDir:
    def test_finds_project_root(self):
        """Should find pyproject.toml and return tmp/qpe_ir_cache."""
        result = default_cache_dir()
        assert result.name == "qpe_ir_cache"
        assert result.parent.name == "tmp"
        # Should be under a directory containing pyproject.toml
        assert (result.parent.parent / "pyproject.toml").exists()


# ==========================================================================
# TestComputeCacheKey
# ==========================================================================


class TestComputeCacheKey:
    def test_deterministic(self, h2_config):
        """Same config produces same key."""
        assert compute_cache_key(h2_config) == compute_cache_key(h2_config)

    def test_different_molecules(self, h2_config, h3o_config):
        """Different molecules produce different keys."""
        assert compute_cache_key(h2_config) != compute_cache_key(h3o_config)

    def test_sensitive_to_estimation_wires(self, h2_config):
        """Changing n_estimation_wires changes the key."""
        config2 = SolvationConfig(
            molecule=h2_config.molecule,
            qpe_config=QPEConfig(n_estimation_wires=5, n_trotter_steps=2),
            verbose=False,
        )
        assert compute_cache_key(h2_config) != compute_cache_key(config2)

    def test_sensitive_to_trotter_steps(self, h2_config):
        """Changing n_trotter_steps changes the key."""
        config2 = SolvationConfig(
            molecule=h2_config.molecule,
            qpe_config=QPEConfig(n_estimation_wires=3, n_trotter_steps=5),
            verbose=False,
        )
        assert compute_cache_key(h2_config) != compute_cache_key(config2)

    def test_insensitive_to_n_waters(self, h2_config):
        """n_waters doesn't affect circuit topology, key should be the same."""
        config2 = SolvationConfig(
            molecule=h2_config.molecule,
            qpe_config=h2_config.qpe_config,
            hamiltonian_mode=h2_config.hamiltonian_mode,
            n_waters=20,
            verbose=False,
        )
        assert compute_cache_key(h2_config) == compute_cache_key(config2)

    def test_insensitive_to_temperature(self, h2_config):
        """Temperature doesn't affect circuit topology."""
        config2 = SolvationConfig(
            molecule=h2_config.molecule,
            qpe_config=h2_config.qpe_config,
            hamiltonian_mode=h2_config.hamiltonian_mode,
            temperature=500.0,
            verbose=False,
        )
        assert compute_cache_key(h2_config) == compute_cache_key(config2)

    def test_includes_molecule_name(self, h2_config):
        """Key should contain molecule name."""
        key = compute_cache_key(h2_config)
        assert "H2" in key

    def test_includes_active_space(self, h2_config):
        """Key should encode active space."""
        key = compute_cache_key(h2_config)
        assert "2e" in key
        assert "2o" in key

    def test_sensitive_to_circuit_style(self, h2_config):
        """Fixed and dynamic modes produce different cache keys (different IR)."""
        config_dynamic = SolvationConfig(
            molecule=h2_config.molecule,
            qpe_config=h2_config.qpe_config,
            hamiltonian_mode="dynamic",
            verbose=False,
        )
        assert compute_cache_key(h2_config) != compute_cache_key(config_dynamic)

    def test_hf_corrected_shares_dynamic_key(self, h2_config):
        """hf_corrected and dynamic share the same cache key (same IR structure)."""
        config_hf = SolvationConfig(
            molecule=h2_config.molecule,
            qpe_config=h2_config.qpe_config,
            hamiltonian_mode="hf_corrected",
            verbose=False,
        )
        config_dyn = SolvationConfig(
            molecule=h2_config.molecule,
            qpe_config=h2_config.qpe_config,
            hamiltonian_mode="dynamic",
            verbose=False,
        )
        assert compute_cache_key(config_hf) == compute_cache_key(config_dyn)


# ==========================================================================
# TestCachePath
# ==========================================================================


class TestCachePath:
    def test_default_directory(self, h2_config):
        """With ir_cache_dir=None, uses default cache dir."""
        path = cache_path_for_config(h2_config)
        assert path.parent == default_cache_dir()

    def test_custom_directory(self, tmp_path):
        """Custom ir_cache_dir is used."""
        config = SolvationConfig(
            molecule=MoleculeConfig(
                name="H2",
                symbols=["H", "H"],
                coords=[[0, 0, 0], [0, 0, 0.74]],
                charge=0,
            ),
            ir_cache_dir=str(tmp_path / "my_cache"),
            verbose=False,
        )
        path = cache_path_for_config(config)
        assert path.parent == tmp_path / "my_cache"

    def test_ll_suffix(self, h2_config):
        """Cache file has .ll extension."""
        path = cache_path_for_config(h2_config)
        assert path.suffix == ".ll"


# ==========================================================================
# TestIsCacheAvailable
# ==========================================================================


class TestIsCacheAvailable:
    def test_no_file(self, tmp_path):
        """No cache file -> not available."""
        config = SolvationConfig(
            molecule=MoleculeConfig(
                name="H2",
                symbols=["H", "H"],
                coords=[[0, 0, 0], [0, 0, 0.74]],
                charge=0,
            ),
            ir_cache_dir=str(tmp_path / "empty_cache"),
            verbose=False,
        )
        assert not is_cache_available(config)

    def test_cache_disabled(self, tmp_path):
        """ir_cache_enabled=False -> not available even if file exists."""
        config = SolvationConfig(
            molecule=MoleculeConfig(
                name="H2",
                symbols=["H", "H"],
                coords=[[0, 0, 0], [0, 0, 0.74]],
                charge=0,
            ),
            ir_cache_dir=str(tmp_path),
            ir_cache_enabled=False,
            verbose=False,
        )
        # Create a dummy cache file
        cp = cache_path_for_config(config)
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_text("dummy IR")
        assert not is_cache_available(config)

    def test_force_recompile(self, tmp_path):
        """ir_cache_force_recompile=True -> not available even if file exists."""
        config = SolvationConfig(
            molecule=MoleculeConfig(
                name="H2",
                symbols=["H", "H"],
                coords=[[0, 0, 0], [0, 0, 0.74]],
                charge=0,
            ),
            ir_cache_dir=str(tmp_path),
            ir_cache_force_recompile=True,
            verbose=False,
        )
        cp = cache_path_for_config(config)
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_text("dummy IR")
        assert not is_cache_available(config)

    def test_valid_cache(self, tmp_path):
        """Valid cache file exists -> available."""
        config = SolvationConfig(
            molecule=MoleculeConfig(
                name="H2",
                symbols=["H", "H"],
                coords=[[0, 0, 0], [0, 0, 0.74]],
                charge=0,
            ),
            ir_cache_dir=str(tmp_path),
            verbose=False,
        )
        cp = cache_path_for_config(config)
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_text("valid LLVM IR content")
        assert is_cache_available(config)


# ==========================================================================
# TestSerializeConfig
# ==========================================================================


class TestSerializeConfig:
    def test_molecule_fields_complete(self, h2_config):
        """Serialized dict should contain all molecule fields."""
        d = _serialize_config_for_subprocess(h2_config)
        mol = d["molecule"]
        assert mol["name"] == "H2"
        assert mol["symbols"] == ["H", "H"]
        assert mol["charge"] == 0
        assert mol["active_electrons"] == 2
        assert mol["active_orbitals"] == 2
        assert mol["basis"] == "sto-3g"

    def test_subprocess_quiet(self, h2_config):
        """Subprocess config should have verbose=False."""
        d = _serialize_config_for_subprocess(h2_config)
        assert d["verbose"] is False

    def test_no_cache_recursion(self, h2_config):
        """Subprocess config should disable caching to prevent recursion."""
        d = _serialize_config_for_subprocess(h2_config)
        assert d["ir_cache_enabled"] is False

    def test_qpe_config_preserved(self, h2_config):
        """QPE parameters should be faithfully serialized."""
        d = _serialize_config_for_subprocess(h2_config)
        qpe = d["qpe_config"]
        assert qpe["n_estimation_wires"] == 3
        assert qpe["n_trotter_steps"] == 2
        assert qpe["n_shots"] == 0

    def test_roundtrip(self, h2_config):
        """Serialize -> reconstruct should produce equivalent config."""
        from q2m3.solvation.ir_cache import _reconstruct_config

        d = _serialize_config_for_subprocess(h2_config)
        config2 = _reconstruct_config(d)
        assert config2.molecule.name == h2_config.molecule.name
        assert config2.qpe_config.n_estimation_wires == h2_config.qpe_config.n_estimation_wires
        assert config2.verbose is False
        assert config2.ir_cache_enabled is False


# ==========================================================================
# Integration Tests (require Catalyst)
# ==========================================================================


@pytest.mark.solvation
class TestCacheIntegration:
    """Integration tests that require Catalyst compilation."""

    def test_cache_miss_then_hit(self, tmp_path):
        """First run writes cache, second run hits cache."""
        from q2m3.solvation import MoleculeConfig, QPEConfig, SolvationConfig, run_solvation

        config = SolvationConfig(
            molecule=MoleculeConfig(
                name="H2",
                symbols=["H", "H"],
                coords=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
                charge=0,
            ),
            qpe_config=QPEConfig(n_estimation_wires=3, n_trotter_steps=2),
            hamiltonian_mode="fixed",
            n_waters=3,
            n_mc_steps=5,
            ir_cache_dir=str(tmp_path / "cache"),
            verbose=False,
        )

        # Run 1: cache miss
        r1 = run_solvation(config, show_plots=False)
        assert not r1["cache_stats"]["is_cache_hit"]

        # Run 2: cache hit
        r2 = run_solvation(config, show_plots=False)
        assert r2["cache_stats"]["is_cache_hit"]

        # Energies should be close (same config, same seed)
        assert abs(r1["best_energy"] - r2["best_energy"]) < 1e-6

    def test_force_recompile_bypasses_cache(self, tmp_path):
        """force_recompile=True should recompile even if cache exists."""
        from q2m3.solvation import MoleculeConfig, QPEConfig, SolvationConfig, run_solvation

        config = SolvationConfig(
            molecule=MoleculeConfig(
                name="H2",
                symbols=["H", "H"],
                coords=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
                charge=0,
            ),
            qpe_config=QPEConfig(n_estimation_wires=3, n_trotter_steps=2),
            hamiltonian_mode="fixed",
            n_waters=3,
            n_mc_steps=5,
            ir_cache_dir=str(tmp_path / "cache"),
            verbose=False,
        )

        # Run 1: populate cache
        r1 = run_solvation(config, show_plots=False)
        assert not r1["cache_stats"]["is_cache_hit"]

        # Run 2: force recompile
        config_force = SolvationConfig(
            molecule=config.molecule,
            qpe_config=config.qpe_config,
            hamiltonian_mode="fixed",
            n_waters=3,
            n_mc_steps=5,
            ir_cache_dir=str(tmp_path / "cache"),
            ir_cache_force_recompile=True,
            verbose=False,
        )
        r2 = run_solvation(config_force, show_plots=False)
        assert not r2["cache_stats"]["is_cache_hit"]

    def test_cache_disabled_no_file(self, tmp_path):
        """ir_cache_enabled=False should not create cache files."""
        from q2m3.solvation import MoleculeConfig, QPEConfig, SolvationConfig, run_solvation

        config = SolvationConfig(
            molecule=MoleculeConfig(
                name="H2",
                symbols=["H", "H"],
                coords=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
                charge=0,
            ),
            qpe_config=QPEConfig(n_estimation_wires=3, n_trotter_steps=2),
            hamiltonian_mode="fixed",
            n_waters=3,
            n_mc_steps=5,
            ir_cache_dir=str(tmp_path / "cache"),
            ir_cache_enabled=False,
            verbose=False,
        )

        r = run_solvation(config, show_plots=False)
        assert not r["cache_stats"]["is_cache_hit"]
        # No cache directory should be created
        assert not (tmp_path / "cache").exists()
