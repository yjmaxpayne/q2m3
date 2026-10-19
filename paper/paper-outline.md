<!--
q2m3 — JOSS paper.md OUTLINE / SPEC
Status: structure approved 2026-05-26; repositioned 2026-05-26 (capability-panorama);
        restructured 2026-05-26 (capability-triad headline, provenance demoted to a
        reproducibility property, grounded in codegraph + autoresearch results).
Purpose: source-of-truth spec for the eventual paper/paper.md + paper.bib + figures.
This file is NOT the submission. It maps every planned paragraph to its source so the
final prose stays faithful and introduces no claim not already backed by an artifact.
Source priority for a *software* paper (code first, slides last):
  1. SHIPPING CODE (ground truth for capability claims): src/q2m3/** — verified via codegraph.
     Real public API: q2m3.core.estimate_resources, QPEEngine, QuantumQMMM;
     q2m3.interfaces.build_fixed_mo_embedding_integrals; q2m3.solvation.run_solvation.
     Real mode names: hf_corrected / fixed / dynamic (config.py:120).
  2. AUTORESEARCH RESULTS (display-value artifacts, with provenance):
     ~/Documents/Papers/paper_drafts/jctc_q2m3_2026/experiments/{S2-01,S3-01,E10-01,E10-02}/...
     + source_bundles/q2m3_output/{qre_survey.csv, ir_qre_correlation_D*.png, h3o_8bit_qpe_benchmark*}
     + reproduce_all_figures.sh (one-command regeneration).
  3. 04-09 slides + diagrams (dev/ref_docs/slides/progress/): narrative + figure source for Fig1/Fig2.
  4. JCTC manuscript (submission/sections/*.tex): related-work framing + citation source only.
GROUND-TRUTH CORRECTIONS from codegraph (do NOT repeat slide overclaims):
  - "8-bit feasibility" = COMPILE feasibility (does it fit memory / MLIR), NOT a surface-code cost model.
  - shot-noise mode exists in the circuit but is NOT wired into the MC loop; iterative QPE is in a
    docstring only, NOT implemented (QPE is single-shot inverse-QFT). Do not claim either as shipping.
  - HF state prep uses X gates, not qml.BasisState (Catalyst ctrl bug).
DO NOT HEADLINE (confounded/shaky): δ_corr-pol (Trotter-depth confound, flips sign); QPE–HF "gap" as
chemistry; 6-31G dynamic (OOM-boundary record only). These stay in the companion manuscript.
-->

# q2m3 — JOSS `paper.md` outline

## JOSS constraints applied (from joss.readthedocs.io, fetched 2026-05-26)

- Length **750–1750 words**; target **~1300**. Full-length / API docs NOT permitted (API lives in software docs).
- Paper **must NOT focus on new research results** — this is a *software* paper. The bounded/null finding is background context, not the headline.
- **Required sections, in order:** Summary · Statement of need · State of the field · Software design · Research impact statement · AI usage disclosure · Acknowledgements · References.
- Citations: `paper.bib` (BibLaTeX/BibTeX); cite with `[@key]` or `@key`; full venue names (no abbreviation-only).
- Figures: captioned image alone in a paragraph; reference with `\autoref{label}` (label in caption).

## Pre-review gate status (verified against `~/Dev/q2m3`)

- **Public dev history >6 mo + iterative** — PASS: 141 commits over 7 months (2025-10-23 → 2026-05-26), distributed Oct/Nov/Jan/Feb/Mar/Apr/May; tags v0.1.0/0.1.1/0.1.2. GitHub repo public since 2025-10-23 (author-confirmed), so public history ≥6 months — gate satisfied.
- **Substantial scholarly effort** — PASS: 9719 LOC across 6 modules (solvation 3694, core 2515, profiling 1317, interfaces 1136, sampling 574, utils 244); 38 test files; CI (ci/docs/publish/release-drafter). Not a thin wrapper.
- **Demonstrated research impact** — strategy: a capability triad backed by reproducibility (14-molecule EFTQC resource survey; IR↔QRE compile-scaling fits; 4/6/8-bit QPE feasibility with graceful fallback), one-command figure regeneration (`reproduce_all_figures.sh`), Zenodo DOI, and a companion methods preprint as supporting anchor (see §5).
- **Open-source practice gaps to close before submission (repo work, NOT part of paper.md):** add `CONTRIBUTING.md`; add a support/governance statement (README or docs); add JOSS draft-paper GitHub Action (`openjournals/inara`).

## Author metadata (resolved 2026-05-26)

1. **Author affiliation** — *Independent* (author-confirmed; supersedes the A\*STAR line in the JCTC manuscript, matches the Zenodo record).
2. **ORCID** — `0000-0003-1963-0865` (author-confirmed).
3. **Corresponding email** — `yjmaxpayne@hotmail.com` (author-confirmed; entered in the JOSS submission form, not in `paper.md` frontmatter).
4. **Companion preprint** — not yet deposited; to be put on arXiv and cited at submission time (see §5).

---

## Frontmatter (YAML) plan

```yaml
---
title: 'q2m3: A reproducible hybrid QC-QM/MM workflow with three QPE solvation modes and matched EFTQC resource ledgers'
tags:
  - Python
  - quantum chemistry
  - quantum computing
  - QM/MM
  - quantum phase estimation
  - EFTQC
  - resource estimation
  - reproducibility
authors:
  - name: Jun Ye
    orcid: 0000-0003-1963-0865
    affiliation: 1
affiliations:
  - index: 1
    name: Independent
date: '[on submission]'
bibliography: paper.bib
---
```

---

## Section-by-section outline

### 1. Summary  (~130 words; non-specialist, avoid jargon)
**Source:** `abstract.tex:1–9`; `introduction.tex:1–24` (§1.1 EFTQC for quantum chemistry);
shipping code `q2m3/{core,interfaces,solvation}`; 04-09 slides `s01_recap` / `s02_solvation`.
**Say:** q2m3 is an open-source Python package that couples molecular electronic-structure preparation (PySCF) with quantum-phase-estimation (QPE) and double-factorization (DF) resource estimation (PennyLane / Catalyst) to study explicitly-solvated molecules on early fault-tolerant quantum computers (EFTQC). One call (`q2m3.core.estimate_resources`) turns a molecule + active space into an EFTQC resource ledger (Hamiltonian 1-norm, logical qubits, Toffoli count); `q2m3.solvation.run_solvation` runs a Monte-Carlo solvation trajectory in one of **three QPE modes** (`hf_corrected` / `fixed` / `dynamic`), where the `dynamic` mode evolves a runtime-coefficient Hamiltonian inside a single compiled circuit, avoiding per-configuration recompilation. Every reported figure is regenerable from one script.
**Citations:** `[@Reiher2017FeMoco; @Berry2019Qubitization; @Kshirsagar2024EFTQC; @JCTC2025Perspective]`.
**Guard:** no numbers/results here; functionality + purpose only. Mode names and "regenerable" are *capabilities*, not research results. Provenance/auditability is **one clause at the end, not a co-headline** — the headline is the workflow + the three modes.

### 2. Statement of need  (~270 words)
**Source:** `introduction.tex:63–117` (§1.2 explicit solvation); `:141–192` (§1.3 contribution & scope); 04-09 slides `s02_solvation` (engineering-gap bullet).
**Say:**
- Environment-induced shifts rival correlation effects, but most quantum-solvation workflows embed the environment as a continuum/distribution, not explicit neighbouring molecules — a gap when the target depends on discrete H-bond geometry or asymmetric ion-solvation shells. (`:63–82`)
- EFTQC resource studies typically start from *vacuum* electronic Hamiltonians and reuse gas-phase resource tables; there is no tested, public tool that carries an **explicit-MM one-electron perturbation** through a fixed-MO / fixed two-electron-tensor frame into **matched DF resource ledgers + finite-depth QPE diagnostics** under one provenance record. (`:93–117`)
- **Engineering gap (new, from 04-09 slides `s02`):** workflows that *do* couple QPE to an explicit-solvent Monte-Carlo loop face a hard choice — freeze a single vacuum Hamiltonian (losing per-configuration electronics) or pay full Catalyst/XLA recompilation at every solvent move. q2m3 demonstrates a third path: passing Hamiltonian coefficients as JAX-traced arrays (`qml.dot(coeffs, OPS)` + `TrotterProduct(check_hermitian=False)`) so one compiled QPE circuit is reused across configurations, making per-step QPE inside an MC loop tractable.
- Target audience: EFTQC/quantum-chemistry resource-estimation practitioners who need auditable, regenerable bookkeeping, **and** practitioners running explicit-solvent QPE sampling who need a working dynamic-Hamiltonian loop — built *on* PySCF + PennyLane, not another solver.
**Citations:** `[@Castaldo2022PCMVQE; @Yoshida20243DRISM; @Merz2025SQD; @QREChem2023; @TFermion2022]`.

### 3. State of the field  (~250 words; explicit build-vs-contribute)
**Source:** `results/05_workflow_comparison.tex:1–11` (workflow-comparison framing); `introduction.tex:93–110` (related resource/embedding work).
**Say:**
- Position against PCM-VQE, 3D-RISM-VQE, IEF-PCM-SQD (implicit/continuum environment); QREChem, TFermion (vacuum resource workflows); DMET / FMO-VQE / multilayer embedding (fragmentation, complementary).
- **Unique contribution (a capability triad, not a new method):** (a) a one-call EFTQC **resource-estimation survey** that runs across many small molecules and active spaces (DF λ / logical qubits / Toffoli); (b) a **runtime-coefficient `dynamic` mode** that runs QPE per solvent configuration inside a single compiled circuit (no per-step recompilation); and (c) **compile-feasibility diagnostics** — measured IR↔QRE compile-cost scaling and a 4/6/8-bit QPE feasibility boundary with graceful fallback. Reproducibility (one-command regeneration + recorded provenance) is a *property of all three*, not a separate headline. Scope, plainly: q2m3 composes existing methods — not a new DF variant, QPE algorithm, or embedding theory — and builds *on* PySCF + PennyLane + Catalyst.
- **Build-vs-contribute:** the comparators target solvation accuracy or fragmentation; none ship a single tool that takes an arbitrary small molecule to a reproducible EFTQC resource ledger *and* runs a dynamic-Hamiltonian QPE solvation loop on the same target, so a standalone package is the smallest useful unit.
**Citations:** `[@Castaldo2022PCMVQE; @Yoshida20243DRISM; @Merz2025SQD; @QREChem2023; @TFermion2022; @Izsak2023Multilayer; @Rossmannek2023DMET; @FMO2024VQE]`.

### 4. Software design  (~430 words; **Figures 1–3**)
**Source:** shipping code (codegraph-verified): `q2m3/core/{qpe.py,resource_estimation.py}`, `q2m3/interfaces/{pyscf_pennylane.py,fixed_mo_embedding.py}`, `q2m3/solvation/{config.py:120,orchestrator.py,circuit_builder.py,energy.py,ir_cache.py}`; `methods.tex:5–78,156–223`; `declarations.tex:98–127`; 04-09 diagrams `solvation_architecture_2026_04_09`, `qpe_driven_workflow_2026_02_25`.
**Say:**
- Architecture: single `src/q2m3/` layout, 6 sub-modules (`core`, `interfaces`, `solvation`, `sampling`, `profiling`, `utils`). Two entry points carry the paper: **`q2m3.core.estimate_resources(...)`** (molecule + active space → `EFTQCResources`: 1-norm, logical qubits, Toffoli count, embedding diagnostics, via PennyLane `DoubleFactorization`) and **`q2m3.solvation.run_solvation(config)`** (a Metropolis MC trajectory with QPE energy evaluation). `q2m3.interfaces.build_fixed_mo_embedding_integrals` builds the MM one-electron embedding. **\autoref{fig:architecture}** (Fig 1).
- Design choice 1 — **three QPE solvation modes** (`config.py:120`, code-true names): `hf_corrected` (per-step QMMM-RHF energy; QPE only as a diagnostic), `fixed` (vacuum Hamiltonian compiled once, zero-argument circuit), and **`dynamic`** (per-configuration MM-embedded Hamiltonian, runtime coefficients). The `dynamic` mode is the distinctive capability — it runs QPE every MC step on a single compiled circuit, resolving the recognised "recompile on every solvent move" blocker. **\autoref{fig:dynamic}** (Fig 2: two-phase compile-once / run-many).
- Design choice 2 — **runtime-coefficient compilation (how `dynamic` works):** the Pauli-operator *structure* is captured once as a closure constant; per step, only the scalar coefficient array changes (`energy.py` patches the identity + Z coefficients from `Δh_MM` in the MO basis) and is passed as a JAX array, reassembled with `qml.dot(coeffs, ops)` inside the `@qjit` circuit. `TrotterProduct(..., check_hermitian=False)` is required so JAX tracers don't trip the hermiticity check (`circuit_builder.py:333,347`). LLVM-IR caching (`ir_cache.py`) reuses the compiled stage across runs of the same topology.
- Design choice 3 — **fixed-MO / fixed-`g` boundary** (`h' = h + Δh_MM^MO`, `g' = g`): a documented modelling boundary that keeps the operator structure invariant across configurations — the precondition that makes Design choice 2 valid; diagonal vs full-`Δh_pq` embedding are selectable in `fixed` mode.
- **Honest capability envelope:** "8-bit feasibility" means *compilation* feasibility (does the circuit fit memory / clear the MLIR→LLVM stage), not a surface-code cost model. `dynamic` compile RSS scales as `n_est × n_trotter × n_terms`; H3O+ 8-bit reproducibly SIGKILLs in Catalyst Phase B (upstream `catalyst#2759`) and falls back to 6-bit. Shot-noise mode exists in the circuit but is not wired into the MC loop; QPE is single-shot inverse-QFT (no iterative refinement). HF state prep uses X gates, not `qml.BasisState`.
- Engineering: CI (`ruff`/`black`/`pytest`/Sphinx-warnings-as-errors), `setuptools_scm` versioning, public API covered by tests. **\autoref{fig:irqre}** (Fig 3: measured IR↔QRE compile-scaling).
**Citations:** `[@PySCF2018; @PySCF2020; @PennyLane2022; @Ittah2024Catalyst; @Berry2019Qubitization; @Senn2009QMMM]`.

### 5. Research impact statement  (~210 words; **Figure 4**; companion route)
**Source:** autoresearch artifacts `experiments/{S2-01,S3-01,E10-01,E10-02}/results/processed/`; `source_bundles/q2m3_output/{qre_survey.csv, ir_qre_correlation_D*.png, h3o_8bit_qpe_benchmark*}`; `reproduce_all_figures.sh`; `declarations.tex:5–71`.
**Say:**
- Realized significance, specific and non-aspirational — **load-bearing legs are the demonstrated capabilities + reproducibility**: (a) a **14-molecule EFTQC resource survey** through one API (CH4/NH3/NH4+/H2O/LiH/H3O+/formamide…, STO-3G and 6-31G, active spaces; 13/14 succeed) — a reusable resource ledger others can extend; (b) **measured IR↔QRE compile-cost scaling** with fitted frontiers (linear OLS, R²≈0.94 `fixed` / 0.98 `dynamic`, 95% CI), turning "will it compile" into a predictable function of problem size; (c) a **4/6/8-bit QPE feasibility boundary** with a documented MLIR-OOM cause and graceful 6-bit fallback; and (d) a **converged QPE anchor** — vacuum H2, 12 estimation bits → dominant-bin error 0.52 mHa (0.33 kcal/mol), inside chemical accuracy — evidence the real circuit pipeline converges (not HF mimicry).
- Reproducibility: every main figure regenerates from `reproduce_all_figures.sh` against bundled processed payloads (`DRY_RUN`/`FIGURES_ONLY`), with recorded env SHA256, git commit, wall time and RSS per result; archived at Zenodo version DOI 10.5281/zenodo.20115477 (concept 10.5281/zenodo.20114945).
- Companion methods study: *supporting* anchor only — *in preparation / nearing submission* (arXiv, cited at submission), which JOSS permits.
- **\autoref{fig:feasibility}** (Fig 4): 4/6/8-bit QPE compile-feasibility with the MLIR-OOM boundary and graceful fallback — a software-capability map, not a chemistry result.
**Citations:** `[@<companion-preprint TOADD>]` + Zenodo software DOI.
**Guard:** all figures are *software output / capability*, never chemistry claims. `δ_corr-pol`, the QPE–HF "gap", and 6-31G dynamic numbers are **excluded** (confounded / boundary-only) — they live in the companion manuscript.

### 6. AI usage disclosure  (~80 words; new, required)
**Say (honest, specific):** Generative-AI assistance (Anthropic Claude / Claude Code, with an internal iterative "autoresearch" orchestration) was used for code scaffolding, experiment orchestration, documentation, and manuscript/paper drafting. All core design decisions (the three-mode abstraction, the runtime-coefficient `dynamic` mode, the fixed-MO / fixed-`g` boundary, API design, experiment design) were made by the human author, who reviewed, edited, and validated all AI-assisted outputs; correctness is enforced by the test suite and CI. State tool versions where known.

### 7. Acknowledgements  (~30 words)
**Source:** `declarations.tex:156–160`, `:135–141`. PennyLane & Catalyst open-source communities; no funding; no competing interests.

### 8. References
`paper.bib` = relevant subset of `submission/references.bib` (keys below) + companion preprint + software citations (PySCF, PennyLane, Catalyst, Zenodo).

---

## Figure plan (4 total: 2 new SVG from slide diagrams + 2 from autoresearch results) — `[user-approved 2026-05-26]`

| Fig | label | type | content | source |
|-----|-------|------|---------|--------|
| 1 | `fig:architecture` | **new SVG** (redraw of `solvation_architecture_2026_04_09`) | `q2m3.solvation` architecture: Config → Circuit Builder (+ LLVM-IR cache) → Energy → MC Loop (Metropolis) → Orchestrator (`run_solvation`), with the three-mode branch (`hf_corrected`/`fixed`/`dynamic`); plus the resource path `estimate_resources` → `DoubleFactorization` ledger. | diagram `diagrams/solvation_architecture_2026_04_09.tex`; code `q2m3/solvation/*`, `q2m3/core/resource_estimation.py` |
| 2 | `fig:dynamic` | **new SVG** (redraw of `qpe_driven_workflow_2026_02_25`) | `dynamic` mode two-phase design: Phase 1 (one-time) build `H_eff` template → `@qjit` precompile QPE with runtime coeffs (`check_hermitian=False`) → compiled closure; Phase 2 (per MC step) propose `R_MM` → fused PySCF callback (coeffs) → precompiled QPE call → Metropolis. Emphasis: **compile once / run many**. | diagram `diagrams/qpe_driven_workflow_2026_02_25.tex`; code `circuit_builder.py:333,347`, `energy.py` |
| 3 | `fig:irqre` | autoresearch result (existing PNG, recolour to CB-safe) | Measured **IR↔QRE compile-cost scaling**: compile RSS / IR-op count vs QRE metric, OLS fit with 95% CI for `fixed` (R²≈0.94) and `dynamic` (R²≈0.98). | `source_bundles/q2m3_output/ir_qre_correlation_D2_*.png` (+ `D4`); `experiments/S3-01_*/results/processed/figure1_fit_diagnostics.json` |
| 4 | `fig:feasibility` | autoresearch result (regenerate from data) | **4/6/8-bit QPE compile-feasibility**: 4-bit & 6-bit succeed (RSS / wall time, 6-bit = 4× resolution), 8-bit SIGKILL in Catalyst Phase B (MLIR→LLVM, `catalyst#2759`) with graceful 6-bit fallback. | `source_bundles/q2m3_output/h3o_8bit_qpe_benchmark_report.md` (+ `.json`, `_6bit.json`); `experiments/E10-02_*/results/processed/summary.md` |

**Dropped:** former `fig:workflow` (abstract provenance dataflow) — provenance is demoted, no longer needs a dedicated figure; former `fig:envelope`/`fig:capability`/`fig:hbond` superseded by the autoresearch-grounded Fig 3 & 4. The H-bond audit-delta and `figure1_trotter_solvation` stay in the companion paper.

**Generation (implementation phase):** redraw Fig 1 & 2 deterministically as SVG from the named TikZ diagrams (graphviz/mermaid or hand-written SVG; colourblind-safe). Fig 3: recolour the bundled `ir_qre_correlation_D2`/`D4` PNG, or regenerate via `reproduce_all_figures.sh`. Fig 4: regenerate from the 8-bit benchmark JSON. **Verify every number against the bundled JSON/CSV before drawing — do not trust slide values blind.** (Fig 3 fits R²≈0.94/0.98 and Fig 4 RSS/wall figures must come from `S3-01`/`E10-02` artifacts.)

---

## `paper.bib` plan

**Pull from `submission/references.bib` (verified present):**
`Reiher2017FeMoco, Berry2019Qubitization, Kshirsagar2024EFTQC, JCTC2025Perspective, Castaldo2022PCMVQE, Yoshida20243DRISM, Merz2025SQD, QREChem2023, TFermion2022, Izsak2023Multilayer, Rossmannek2023DMET, FMO2024VQE, PySCF2018, PySCF2020, PennyLane2022, Senn2009QMMM` (+ optionally `Lee2021THC, Childs2021Trotter` if cited).

**To add:**
- Companion methods preprint (arXiv) — `[CONFIRM arXiv id]`.
- Software citations with full venue names: PySCF (`PySCF2018`/`PySCF2020`), PennyLane (`PennyLane2022`), **PennyLane-Catalyst** (`Ittah2024Catalyst` — Ittah et al., *JOSS* 9(99), 6720, cited in §4), and the q2m3 Zenodo software DOI.
- Optional, if the upstream blocker is cited in §4/§5: a footnote/URL to `PennyLaneAI/catalyst#2759` (issue link, not a bib entry).

---

## Word budget (target ~1290; ceiling 1750)

| Section | Words |
|---|---|
| Summary | 130 |
| Statement of need | 270 (+ engineering-gap bullet) |
| State of the field | 250 (capability triad) |
| Software design | 430 (3 design choices + honest envelope; Figs 1–3) |
| Research impact | 210 (capability triad + reproducibility; Fig 4) |
| AI usage disclosure | 80 |
| Acknowledgements | 30 |
| **Total** | **~1400** (ceiling 1750; comfortable) |

## JOSS editor review (risks & actions) — `[self-review as handling editor, 2026-05-26]`

Reviewed against the current (2024-tightened) JOSS criteria: `joss.readthedocs.io/en/latest/review_criteria.html`.
Verdict: structure is compliant and the honesty bar is high, but two **desk-rejection-level** risks
must be cleared before submission. The core risk is not prose — it is the *research-grade reuse
evidence chain* for the software itself.

### 🔴 Blocking
- **B1 — "Substantial scholarly effort" / reuse evidence (make-or-break).** JOSS now requires a
  "meaningful contribution to the research community **rather than being a one-off tool for a single
  analysis**", with acceptable signals being published research citing the software, adoption by other
  groups, or workflow integration. q2m3 is single-author, niche, with no external adoption, and the
  companion is unpublished. **Worse, the repo self-describes as a "Functional POC" — the editor WILL
  read the repo.** Action: (a) de-POC the repo (README/CLAUDE.md) to a research-grade framing before
  submission; (b) strengthen §5 reuse evidence beyond "reproducibility + unpublished companion"
  (user-facing tutorial/notebook, reachable API docs, any actual third-party/teaching use).
- **B2 — Reproduction artifacts live OUTSIDE the reviewed repo.** A JOSS reviewer clones the software
  repo and runs it. But `reproduce_all_figures.sh`, `experiments/{S2-01,S3-01,E10-01,E10-02}/`,
  `qre_survey.csv`, `ir_qre_correlation_D*.png` are in `~/Documents/Papers/paper_drafts/jctc_q2m3_2026/`
  — a separate tree; only `paper.md` lives on the `paper/` branch. The §5 "every figure regenerates
  from one script" claim is then **not executable from the reviewed repo**. Action: either migrate the
  figure-regeneration scripts + processed payloads into the q2m3 repo (`paper/` or `examples/reproduce/`),
  or point `paper.md` at a public, reviewer-accessible archive (Zenodo / standalone repo) and confirm reach.

### 🟠 Major
- **M1 — "Install and verify core functionality" hits the toy-scale ceiling.** The headline `dynamic`
  mode is memory-blocked for H3O+, SIGKILLs at 8-bit, OOMs at `n_trotter ≥ 7`. A reviewer running a
  non-trivial system will hit the wall. Honesty (§4 envelope) ≠ expectation management. Action: state
  the **reliable operating envelope** positively in §2/§4 (which sizes/depths run stably) and ship a
  minimal example guaranteed to run on a commodity machine.
- **M2 — Summary/§3 oversell `dynamic`; scale caveats quarantined in §4.** Move a half-sentence scale
  limit into §3 unique-contribution (b) so the headline and the honest boundary sit together.
- **M3 — State of the field compares to methods/papers, not software packages.** JOSS wants comparison
  to "other commonly-used **packages**". PCM-VQE/3D-RISM-VQE/QREChem/TFermion are methods. Action: name
  1–2 real comparable software packages (e.g. OpenFermion, Qiskit Nature, PennyLane qchem, Tangelo) and
  locate q2m3 in the software ecosystem.
- **M4 — Documentation editor-checklist items.** Beyond CONTRIBUTING/governance (already tracked),
  confirm: automated install, **user-facing example usage solving a real problem**, reachable API docs,
  CI-backed tests.

### 🟡 Minor
- **m1** — §2 contains implementation detail (`qml.dot`/`check_hermitian=False`); Statement of need
  should be problem + audience only. Remove it from §2 (it already lives in §4).
- **m2** — Title is long/jargon-dense ("matched EFTQC resource ledgers"). Consider a shorter title.
- **m3** — Keep §4 from drifting into a methods paper (drop derivations like `h'=h+Δh_MM^MO`; point to docs).
- **m4** — Word count not officially pinned; ~1400 is acceptable under the new multi-section structure,
  but §4/§5 are the overrun risks — keep a *software*, not *research*, voice.
- **m5** — Do not let "Research impact" rest on an unpublished companion (already demoted; see B1).

### ✅ Done right (keep)
- Five required sections in the correct 2024-tightened order.
- Correct exclusion of confounded science (δ_corr-pol, QPE–HF gap) — editors value this.
- codegraph-grounded honesty (8-bit = compile feasibility; shot-noise not wired; X-gate prep) protects
  you when the reviewer runs the code.

## Pre-submission checklist (repo work, not part of paper.md)

**Blocking (clear first):**
- [ ] **B1a** — De-POC the repo: rewrite README/`CLAUDE.md` "Functional POC" framing to research-grade.
- [ ] **B1b** — Add a user-facing tutorial/notebook + reachable API docs; capture any third-party/teaching use as reuse evidence for §5.
- [ ] **B2** — Decide reproduction-artifact home: migrate `reproduce_all_figures.sh` + processed payloads into the q2m3 repo, OR point `paper.md` at a public reviewer-accessible archive; verify a reviewer can regenerate every figure from the reviewed repo.

**Major:**
- [ ] **M1** — State the reliable operating envelope in §2/§4; ship a minimal example that runs on a commodity machine.
- [ ] **M2** — Add the scale-limit half-sentence to §3 (b).
- [ ] **M3** — Name 1–2 comparable software packages in §3 (not just methods/papers).
- [ ] **M4** — Verify automated install, example usage, reachable API docs, CI-backed tests; add `CONTRIBUTING.md` + support/governance statement.

**Standard:**
- [ ] Confirm GitHub repo public-visibility ≥6 months (gate).
- [ ] Add JOSS draft-paper GitHub Action (`openjournals/inara`) and verify `paper.pdf` compiles.
- [ ] Generate figures: `fig:architecture` + `fig:dynamic` as SVG (redraw from the named TikZ diagrams); `fig:irqre` (recolour/regenerate IR↔QRE PNG) + `fig:feasibility` (regenerate from 8-bit benchmark JSON) — verify all numbers against bundled `S3-01`/`E10-02` artifacts before drawing.
- [ ] Deposit companion methods manuscript on arXiv; insert its id in §5 + `paper.bib`.
- [ ] Fill `[CONFIRM]` frontmatter (affiliation, ORCID, email).
- [ ] **m1/m2/m3** — Trim §2 implementation detail; shorten title; keep §4 out of methods-paper voice.
- [ ] Draft `paper/paper.md` prose from this outline (writing-plans phase); software (not research) voice.
