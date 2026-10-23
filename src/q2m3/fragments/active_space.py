"""Active-space selection for counterpoise subsystems: AVAS labels, ghost guard, preflight.

AVAS searches its labels on a MINAO copy of the molecule, as regular expressions over the
AO labels, where ghost atoms appear as ``GHOST-<element>``. Labels are therefore resolved
here to one anchored pattern per real-atom AO of that copy.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

import numpy as np
from pyscf import gto
from pyscf.mcscf import avas

from q2m3.fragments.counterpoise import Subsystem
from q2m3.fragments.errors import FragmentWorkflowError
from q2m3.fragments.result import ActiveSpaceRecord

MAX_NORB = 10


def _minao_copy(mol: gto.Mole) -> gto.Mole:
    # Same copy as pyscf.mcscf.avas builds before it searches the labels.
    pmol = mol.copy()
    pmol.atom, pmol.unit, pmol.symmetry, pmol.basis = mol._atom, "B", False, "minao"
    pmol.build(False, False)
    return pmol


def _parse(label: str, aos: list[tuple]) -> tuple[frozenset[int], str]:
    parts = label.split()
    atom = None
    if len(parts) == 3 and parts[0].isdigit():
        atom, parts = int(parts[0]), parts[1:]
    if len(parts) != 2:
        raise ValueError(f"AVAS label {label!r} is not '<i> <Elem> <shell>' or '<Elem> <shell>'")
    element, shell = parts
    atoms = frozenset(
        a for a, symbol, nl, _m in aos if atom in (None, a) and symbol == element and nl == shell
    )
    if not atoms:
        raise ValueError(f"AVAS label {label!r} matches no MINAO AO of the molecule")
    return atoms, shell


def resolve_labels(
    subsystems: Mapping[str, Subsystem], labels: Sequence[str]
) -> dict[str, tuple[str, ...]]:
    """Resolve AVAS labels to anchored AO patterns for each subsystem.

    Labels refer to the full geometry: ``"<i> <Elem> <shell>"`` is atom ``i`` (0-based),
    ``"<Elem> <shell>"`` every atom of that element. A subsystem keeps the AOs of its
    real atoms only, as patterns ``^<AO label>\\s*$`` over its MINAO copy.

    Raises:
        ValueError: A label is malformed or matches no AO of the full molecule.
        FragmentWorkflowError: A subsystem keeps no label because all are on ghost atoms.
    """
    reference = subsystems["AB"] if "AB" in subsystems else subsystems["M"]
    reference_aos = _minao_copy(reference.mol).ao_labels(fmt=False)
    targets = [_parse(label, reference_aos) for label in labels]
    patterns = {}
    for name, sub in subsystems.items():
        pmol = _minao_copy(sub.mol)
        text = pmol.ao_labels()
        real = set(sub.real_atoms)
        kept = [
            "^" + re.escape(text[i].strip()) + r"\s*$"
            for i, (a, _symbol, nl, _m) in enumerate(pmol.ao_labels(fmt=False))
            if a in real and any(a in atoms and nl == shell for atoms, shell in targets)
        ]
        if not kept:
            raise FragmentWorkflowError(
                f"subsystem {name} keeps no AVAS label: {list(labels)} are all on ghost atoms"
            )
        patterns[name] = tuple(kept)
    return patterns


def minao_indices(subsystem: Subsystem, ao_labels: Sequence[str]) -> tuple[int, ...]:
    """MINAO AOs that AVAS selects with ``ao_labels``; none may belong to a ghost atom.

    Raises:
        FragmentWorkflowError: A selected AO belongs to a ghost atom.
    """
    pmol = _minao_copy(subsystem.mol)
    selected = tuple(int(i) for i in pmol.search_ao_label(list(ao_labels)))
    atom_of = [a for a, *_ in pmol.ao_labels(fmt=False)]
    ghost = [pmol.ao_labels()[i].strip() for i in selected if atom_of[i] in subsystem.ghost_atoms]
    if ghost:
        raise FragmentWorkflowError(
            f"subsystem {subsystem.name}: AVAS labels select ghost-atom AOs {ghost}"
        )
    return selected


def preflight(records: Mapping[str, ActiveSpaceRecord]) -> None:
    """Reject active spaces that SQD cannot run, before any expensive step.

    Raises:
        FragmentWorkflowError: A subsystem has no active occupied or virtual orbital, an
            odd number of active electrons, or more than ``MAX_NORB`` active orbitals.
    """
    for name, record in records.items():
        occupied = record.nelecas // 2
        if record.nelecas % 2:
            problem = "odd number of active electrons"
        elif occupied == 0:
            problem = "no active occupied orbital"
        elif record.ncas <= occupied:
            problem = "no active virtual orbital"
        elif record.ncas > MAX_NORB:
            problem = f"more than {MAX_NORB} active orbitals"
        else:
            continue
        raise FragmentWorkflowError(
            f"subsystem {name}: ({record.nelecas}e,{record.ncas}o) has {problem}"
        )


def select_active_spaces(
    subsystems: Mapping[str, Subsystem], labels: Sequence[str], threshold: float
) -> dict[str, tuple[ActiveSpaceRecord, np.ndarray]]:
    """Run AVAS with the same labels and threshold for every subsystem.

    All labels are resolved and guarded before the first AVAS run, and the preflight runs
    after the last one. Each subsystem gives its record and the AVAS MO coefficients.

    Raises:
        ValueError: A label is malformed or matches no AO of the full molecule.
        FragmentWorkflowError: Label resolution, the ghost guard or the preflight fails.
    """
    patterns = resolve_labels(subsystems, labels)
    indices = {name: minao_indices(sub, patterns[name]) for name, sub in subsystems.items()}
    spaces = {}
    for name, sub in subsystems.items():
        ncas, nelecas, mo = avas.avas(
            sub.scf, list(patterns[name]), threshold=threshold, canonicalize=True, verbose=0
        )
        nelecas = int(nelecas)
        record = ActiveSpaceRecord(
            nelecas=nelecas,
            ncas=ncas,
            ncore=(sub.mol.nelectron - nelecas) // 2,
            ao_labels=patterns[name],
            minao_indices=indices[name],
        )
        spaces[name] = (record, mo)
    preflight({name: record for name, (record, _mo) in spaces.items()})
    return spaces
