# Changelog

## v0.1.0 (2026-05-08)

### Feat

- **solvation**: add trajectory replay and structure analysis workflows
- **examples**: add standardized IR-QRE trotter-5 analysis
- **examples**: add H3O+ 4-bit/6-bit/8-bit QPE benchmark
- **core**: support active space in QPE resource estimation
- **examples**: add 4-bit vs 8-bit QPE resolution benchmark
- **profiling**: add general-purpose timing utilities module
- **core**: add EFTQC resource estimation module
- **solvation**: add pure-NumPy analysis module for δ_corr-pol
- **utils**: add energy comparison plotting utility
- **solvation**: add Catalyst LLVM IR cache for QPE compilation
- **solvation**: add integration tests and migrate examples to production API
- **solvation**: add orchestrator with end-to-end three-mode MC workflow
- **solvation**: add MC loop with Metropolis acceptance and three-mode energy branching
- **solvation**: add energy computation module with three-mode callback factories
- **solvation**: add QPE circuit builder with parameterized @qjit compilation
- **solvation**: add phase extraction module for QPE energy decoding
- **solvation**: add data layer — config dataclasses and solvent models
- **solvation**: add package skeleton and test infrastructure
- **profiling**: expose public API in profiling/__init__.py
- **profiling**: add orchestrator with subprocess isolation and callback-based progress
- **profiling**: add qpe_profiler with three-phase QPE compilation profiling
- **profiling**: add catalyst_ir module with ir_output_dir and analyze_ir_stages
- **profiling**: add memory measurement module with MemorySnapshot, MemoryTimeline, ParentSideMonitor
- **core**: promote hamiltonian_utils from examples to src/q2m3/core/
- **core**: promote MoleculeConfig from examples to src/q2m3/molecule.py
- **examples**: add H2 cached QPE-driven MC solvation script
- **examples**: add QPE compile cache verification script
- **examples**: add parent-side monitoring and compiler subprocess RSS tracking
- **examples**: add H_fixed mode and IR lifecycle management to QPE profiler
- **examples**: add QPE compilation memory profiler for Catalyst IR analysis
- **orchestrator**: connect qpe_driven mode with statistics label adaptation
- **mc_loop**: add QPE-driven MC loop factory with every-step quantum evaluation
- **energy**: add vacuum cache and fused QPE callback for qpe_driven mode
- **config**: add qpe_driven mode to SolvationConfig
- **examples**: add mm_embedded QPE mode with runtime coefficient architecture
- **examples**: add solvation stabilization and Mulliken charge output to MC workflow
- **analysis**: add Catalyst timing aggregation for Dual-Circuit precompilation
- **examples**: add time statistics breakdown to h2_mc_solvation_qjit
- **examples**: add QJIT pre-compilation example and Catalyst performance guidelines
- **sampling**: add Monte Carlo solvation optimization module
- **profiling**: add fine-grained timing breakdown for QPE bottleneck analysis
- **examples**: integrate profiling to diagnose jit+lightning.gpu bottlenecks
- **examples**: add Catalyst benchmark to H2 QPE minimal example
- added badages to the landing github page of the repo
- added CI/CD workflow for the repo
- add EFTQC resource estimation using PennyLane DoubleFactorization API
- implement MM point charge embedding in QPE Hamiltonian
- add circuit visualization for QPE and RDM using qml.draw
- add RDM GPU acceleration and batched measurement optimization
- add quantum 1-RDM measurement for Mulliken charge analysis
- added lightning.gpu and lightning.qubit support for fast simulation with/without catalyst
- **qpe**: add Catalyst @qjit JIT compilation support
- **interfaces**: implement pyscf_to_pennylane_hamiltonian for molecular Hamiltonian generation

### Fix

- revised release-drafter workflow to avoid draft release on PR
- **solvation**: resolved warnings due to problematic solvent placement
- **solvation**: propagate hf_corrected deferred circuit metadata and IR cache status
- **solvation**: replace deprecated device shots with qml.set_shots transform
- **solvation**: use compile-time constant circuits for fixed mode QPE
- **examples**: restore hamiltonian_utils re-export removed by ruff autofix
- **examples**: fix catastrophic MM energy for large n_waters in cached QPE-MC
- resolve ruff lint errors and consolidate pre-commit hooks
- **examples**: improve scientific accuracy of three-mode QPE output
- **tests**: removed debug only test
- migrate DoubleFactorization API and enhance Catalyst GPU detection
- resolved CI/CD lint errors
- resolved all lint errors
- migrate to qml.set_shots() API for PennyLane v0.43+ compatibility
- resolve Catalyst @qjit BasisState incompatibility in QPE
- resolve Catalyst @qjit BasisState incompatibility in QPE
- correct QPE phase extraction with MSB-first bit ordering
- Catalyst results now correctly report lightning.qubit device
- correct Mulliken charge calculation and improve QPE demo output
- add active space MO-to-AO RDM transformation for Mulliken analysis

### Refactor

- **examples**: archive superseded scripts and restructure README
- **examples**: split h2_qpe_validation into three focused scripts
- **examples**: replace re-export imports with direct q2m3.profiling imports
- **examples**: slim qpe_memory_profile.py from 1555 to 579 lines
- **examples**: updated h2_three_mode_qpe_mc.py example and corresponding module files to enhance description accuracy in output file
- **examples**: fair three-mode comparison with rigorous analysis
- consolidate shared infrastructure into core library
- **examples**: updated naming for all existing examples and revised detail contents for all key readme files
- **example**: modularized mc_solvation and refactored all e2e examples
- **output**: add Catalyst Performance Summary to profiling report
- **examples**: address code review feedback
- **examples**: modularize h3op_qpe demo into package structure
