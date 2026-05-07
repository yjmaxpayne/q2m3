# Architecture

q2m3 is organized around explicit boundaries between classical chemistry,
quantum circuits, MC sampling, and diagnostics. The boundaries matter because
PySCF, NumPy, PennyLane, JAX, and Catalyst each impose different data and
execution constraints.

## Package Overview

```{mermaid}
flowchart TB
    accTitle: q2m3 Package Layers
    accDescr: q2m3 separates orchestration, core quantum chemistry, interfaces, solvation sampling, and diagnostics.

    user["User scripts and examples"]
    core["q2m3.core<br/>QPE, QM/MM, RDM, resources"]
    interfaces["q2m3.interfaces<br/>PySCF to PennyLane bridge"]
    solvation["q2m3.solvation<br/>MC workflow and QPE modes"]
    sampling["q2m3.sampling<br/>Moves, waters, MM force field"]
    profiling["q2m3.profiling<br/>Timing, memory, IR analysis"]
    utils["q2m3.utils<br/>I/O and plotting helpers"]

    user --> core
    user --> solvation
    core --> interfaces
    solvation --> core
    solvation --> sampling
    solvation --> profiling
    core --> profiling
    user --> utils
```

## Data Flow

The main workflow moves from classical molecular data to a qubit Hamiltonian,
through QPE or a fallback solver path, and back to classical observables.

```{mermaid}
sequenceDiagram
    participant User
    participant QMMM as QuantumQMMM
    participant PySCF as PySCF/PennyLane Converter
    participant QPE as QPEEngine
    participant RDM as RDMEstimator

    User->>QMMM: compute_ground_state()
    QMMM->>PySCF: build HF reference and qubit Hamiltonian
    PySCF-->>QMMM: Hamiltonian, HF state, metadata
    QMMM->>QPE: estimate phase and energy
    QPE-->>QMMM: energy, samples/probabilities
    QMMM->>RDM: optional 1-RDM measurement
    RDM-->>QMMM: density matrix and charges
    QMMM-->>User: results dictionary
```

## Module Boundaries

| Package | Responsibility | Boundary rule |
| --- | --- | --- |
| `q2m3.core` | QPE, QM/MM orchestration, RDM, resource estimation, device selection | Owns quantum chemistry algorithms and public computational entry points |
| `q2m3.interfaces` | PySCF/PennyLane conversion and density matrix bridge | Owns framework conversion and MM embedding data exchange |
| `q2m3.solvation` | MC solvation orchestration, Catalyst QPE bundles, IR cache, analysis | Owns compile-once/reuse-many solvation workflows |
| `q2m3.sampling` | Classical water geometry, MC moves, force-field helpers | Keeps MC proposal mechanics independent of quantum execution |
| `q2m3.profiling` | Memory, timing, and Catalyst IR diagnostics | Measures performance without becoming part of the physical model |
| `q2m3.utils` | File I/O and plotting utilities | Keeps low-level helper code out of the algorithm packages |

## Device Selection

Device selection is split across two backend systems:

| Backend family | Used for | Detection surface |
| --- | --- | --- |
| PennyLane devices | Standard QPE execution | `lightning.gpu`, `lightning.qubit`, `default.qubit` |
| JAX/Catalyst backend | Compiled `@qjit` execution | JAX default backend and Catalyst availability |

`device_type="auto"` selects the best available PennyLane device for standard
execution. Catalyst execution still depends on the effective JAX/Catalyst
backend, so a Lightning GPU device name does not by itself prove that compiled
execution is running on the GPU.

## Catalyst And IR Cache

The solvation package treats Catalyst as a compile-once/reuse-many tool. The
QPE circuit topology is compiled once, then fixed or runtime Hamiltonian
coefficients are supplied across MC steps. The IR cache stores Catalyst LLVM IR
for reuse when the structural cache key is unchanged.

The cache key is structural: molecule, basis, active space, QPE wire count,
Trotter depth, and circuit style matter. Runtime solvent coordinates and the
number of waters do not change the circuit topology in the same way.

## Key Design Decisions

| Decision | Reason |
| --- | --- |
| Keep H2 as the first-run path | It exercises real APIs without the compile memory of H3O+ |
| Mark H3O+ and 8-bit workflows as diagnostics | They are useful but can exceed laptop memory limits |
| Use active spaces explicitly | Every energy result needs a clear electron/orbital/qubit context |
| Keep energies in Hartree internally | Unit conversion is visible and avoids mixed-unit mistakes |
| Preserve classical fallback paths | They test QM/MM plumbing when quantum execution is unavailable |
| Separate sampling from quantum execution | MC proposal logic remains testable without Catalyst |

## Extension Points

`q2m3.core.quantum_solver` defines a conservative solver abstraction for future
algorithms such as VQE or QAOA. Current public workflows still route primarily
through `QPEEngine` and `QuantumQMMM`; treat the solver abstraction as an
extension point rather than a fully integrated replacement path.
