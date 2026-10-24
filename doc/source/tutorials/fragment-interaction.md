# Fragment Interaction Energy

This tutorial computes the counterpoise-corrected interaction energy of the
S22 water dimer with one call to `run`. The call returns five energy layers
(HF, MP2, CASCI, SQD, and CCSD(T)), an error decomposition, resource estimates,
and the provenance of the run. `render_report` writes the result as a Markdown
report.

## Requirements

The default mode is `mode="full"`. It needs the optional SQD backend:

```bash
uv sync --frozen --extra sqd
```

The SQD stage runs only on Linux x86_64. On other platforms, use
`mode="classical_only"`. That mode needs only the core install. It returns the
HF and MP2 layers and the active-space records, and it does not run SQD or
CCSD(T).

## Prepare The Geometry

Save the geometry as `water-dimer.xyz`. Coordinates are in Angstrom. Atoms 0-2
are fragment A and atoms 3-5 are fragment B.

```{literalinclude} water-dimer.xyz
:language: text
```

## Run The Workflow

```{testsetup}
import os

# make docs-doctest runs from the repository root; the tutorial reads its XYZ file.
_cwd = os.getcwd()
os.chdir("doc/source/tutorials")
```

```{testcleanup}
os.chdir(_cwd)
```

```{doctest}
>>> from q2m3 import Molecule, run
>>> from q2m3.fragments import render_report
>>> dimer = Molecule.from_xyz("water-dimer.xyz", fragments=[(0, 1, 2), (3, 4, 5)], name="water-dimer")
>>> result = run(dimer, avas_labels=["O 2p", "H 1s"], avas_threshold=0.7)
>>> print(render_report([result]))  # doctest: +ELLIPSIS
# 片段相互作用能报告
...
## 片段 water-dimer（模式 full，基组 cc-pvdz）
...
| HF | -0.005868 | -3.682 | |
| MP2 | -0.006272 | -3.935 | |
| CASCI | -0.005695 | -3.574 | |
| SQD | -0.005669 | -3.557 | |
| CCSD(T) | -0.00589... | -3.699 | |
...
```

The run takes less than one minute on a workstation. The report is in Chinese
and has these sections:

| Section | Content |
| --- | --- |
| Interaction energy | ΔE of each layer in Ha and kcal/mol. An empty cell gives its reason. |
| Error decomposition | Algorithm error (SQD − CASCI), active-space error (CASCI − same-basis full-space reference), basis error, and model error. |
| QPE resources | Double-factorization cost estimate (λ, logical qubits, Toffoli gates) for AB, A, and B. |
| SQD workload | Qubits, full CI dimension, and predicted RSS of the two SQD stages. |

When CCSD(T) runs, its layer is the same-basis full-space reference. The basis
and model errors need an external reference. Give one with
`run(..., references=["refs.csv"])`. The CSV columns are `fragment_id`,
`method`, `basis`, `delta_e`, and `unit` (`Ha`, `kcal/mol`, or `kJ/mol`).
Without an external reference, the two errors are empty and the report gives
the reason.

## Compare With A Hand-Written Route

| Step | Hand-written PySCF route | q2m3 |
| --- | --- | --- |
| Counterpoise SCF | Build AB, A, and B molecules. Mark the partner atoms as `ghost-` atoms. Run three RHF calculations. | `run` |
| MP2 | Count the frozen core orbitals without the ghost atoms. Run three MP2 calculations. | `run` |
| Active space | Run AVAS on each subsystem. Remove the ghost-atom AO labels. Check that the monomer spaces correspond to the dimer space. | `avas_labels`, `avas_threshold` |
| SQD | Build the active-space integrals and their provenance. Run SQD for AB, A, and B with single-threaded math libraries. | `run` |
| CCSD(T) reference | Estimate the memory and wall time. Run frozen-core CCSD(T) when the estimate fits the budget. | `run` |
| Resources | Run the double-factorization estimate on each set of active-space integrals. | `run` |
| Report | Convert the units and write the tables. | `render_report` |

## Choose The Active Space

`avas_labels` and `avas_threshold` have no default values. A label is
`"<Elem> <shell>"` for every atom of an element, or `"<i> <Elem> <shell>"` for
atom `i` (0-based) of the full geometry. The same labels apply to the dimer and
to both monomers.

The active space of each subsystem can have at most 10 orbitals. The PySCF
default threshold 0.2 selects (16e,12o) for the water dimer, so `run` rejects
it. The threshold 0.7 selects (12e,8o) for the dimer and (6e,4o) for each
monomer.

`run` raises `ActiveSpaceMismatchError` when the monomer active spaces do not
correspond to the dimer active space. It raises `FragmentWorkflowError` when an
active space has no active virtual orbital or more than 10 orbitals.

## Reproduce A Result

`result.provenance` records the geometry hash, basis, AVAS labels and
threshold, seed, package versions, and thread settings. To reproduce a result,
run the same geometry again with these values. The deterministic layers and the
SQD energies with the same seed agree to 1e-10 Ha.
