# Figures for the q2m3 JOSS paper

Four figures, each regenerable from a script or a vendored data file. Colours use the
Okabe–Ito colourblind-safe palette (panels reused from the companion manuscript use
`viridis`, which is also colourblind-safe). Diagram PNGs are exported from SVG at 2× via
`cairosvg`.

| File | `paper.md` label | What it shows | Provenance / regeneration |
|------|------------------|---------------|---------------------------|
| `fig1_architecture.{svg,png}` | `fig:architecture` | Two public entry points (`estimate_resources`, `run_solvation`) over a shared PySCF · PennyLane · Catalyst foundation; resource-ledger path and three-mode solvation path. | `make_fig1_architecture.py`. Structure verified against `src/q2m3` (mode names from `solvation/config.py:120`); redraw of `dev/ref_docs/slides/progress/diagrams/solvation_architecture_2026_04_09.tex` with the **code-true** mode names. |
| `fig2_three_modes.{svg,png}` | `fig:dynamic` | The three QPE modes (`fixed` / `hf_corrected` / `dynamic`): what is compiled once, what runs per MC step, the acceptance energy, and δ_corr-pol capture. `dynamic` highlighted (compile-once / run-many). | `make_fig2_three_modes.py`. Timings (compile / per-QPE / total) from `dev/wiki/solvation.md` (H2 + 10 TIP3P, 500 MC steps, verified SSOT); redraw of `qpe_driven_workflow_2026_02_25.tex` extended from one mode to all three. |
| `fig3_irqre_compile_scaling.{png,pdf}` | `fig:irqre` | Peak compile-time RSS vs LLVM-IR op count for `fixed` and `dynamic`; linear fits (fixed R²=0.94, dynamic R²=0.98), ≈28× steeper `dynamic` slope; H₃O⁺ fixed measured (5.7 GB) vs dynamic projected (~101 GB). | `make_fig3_irqre.py` reads `data/irqre_fit_diagnostics.json` (q2m3 **S3-01** artifact). This is the compile-feasibility panel (Panel B) of the companion figure, regenerated standalone. The OLS fit reproduces the artifact exactly. |
| `fig4_catalyst_ion_matrix.{png,pdf}` | `fig:feasibility` | 7-system Catalyst compile matrix: peak RSS + compile time (`fixed` vs runtime-coeff.), with H₂O/H₃O⁺ runtime-coeff. memory boundaries; Catalyst speedup (5× repeated QPE; one-compile/1000-evals). | Rendered by the companion `figure4_catalyst_ion_matrix.py` from `data/ion_catalyst_matrix.csv` (q2m3 **S3-01**, vendored here). Regenerable via `experiments/reproduce_all_figures.sh` (Figure 4). |

## Terminology

The code mode key `dynamic` is shown as **"runtime-coeff."** in `fig:feasibility` (companion-manuscript
label) and as **"dynamic (runtime-coeff.)"** in `fig:irqre`. `paper.md` introduces it once as
"the `dynamic` (runtime-coefficient) mode" so the figures and prose agree.

## Scope note (JOSS policy)

`fig:irqre` deliberately uses **only the compile-feasibility panel** of the companion two-panel
figure. The other panel (a finite-depth δ_corr-pol chemistry cross-section) is a *research result*
and is excluded here: JOSS papers must not focus on new research results, and that panel lives in
the companion methods manuscript. All four JOSS figures are **software-capability** figures, not
chemistry claims.

## Regenerate

```bash
# diagrams (SVG -> PNG); cairosvg pulled in ephemerally, no env pollution
uv run --with cairosvg python make_fig1_architecture.py
uv run --with cairosvg python make_fig2_three_modes.py

# data plot (project .venv: numpy / scipy / matplotlib)
uv run python make_fig3_irqre.py

# fig4 is vendored from the companion submission; regenerate there via
#   experiments/reproduce_all_figures.sh   (Figure 4)
```
