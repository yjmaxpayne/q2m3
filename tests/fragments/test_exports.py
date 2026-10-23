"""Top-level and package exports of the fragment workflow need only the core install."""

from __future__ import annotations

import q2m3
import q2m3.fragments as fragments
from q2m3 import _lazy
from q2m3.fragments import errors, molecule, result, workflow

PACKAGE_EXPORTS = {
    "Molecule": molecule.Molecule,
    "run": workflow.run,
    "FragmentInteractionResult": result.FragmentInteractionResult,
    "SubsystemResult": result.SubsystemResult,
    "FragmentWorkflowError": errors.FragmentWorkflowError,
    "ActiveSpaceMismatchError": errors.ActiveSpaceMismatchError,
    "ReferenceImportError": errors.ReferenceImportError,
}


def test_run_and_molecule_resolve_from_the_top_level_package():
    from q2m3 import Molecule, run

    assert run is workflow.run
    assert Molecule is molecule.Molecule
    assert {"run", "Molecule"} <= set(q2m3.__all__)


def test_core_probe_needs_no_optional_module():
    assert _lazy.EXTRA_PROBES["core"] == ()
    assert _lazy.extra_available("core")


def test_fragments_package_exports_the_workflow_names():
    assert set(fragments.__all__) == set(PACKAGE_EXPORTS)
    for name, value in PACKAGE_EXPORTS.items():
        assert getattr(fragments, name) is value
