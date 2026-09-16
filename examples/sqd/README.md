# SQD dependency replay bundle

These four probes are dependency/integration evidence, not a production SQD engine.
Heavy scientific imports occur only inside probe functions. Run each in a fresh
candidate-environment interpreter, sequentially with OMP/MKL/OPENBLAS threads set to 1:

```bash
python examples/sqd/replay.py --probe h2 --output /path/to/new-run/h2.json
python examples/sqd/replay.py --probe reference --output /path/to/new-run/reference.json
python examples/sqd/replay.py --probe transpose --output /path/to/new-run/transpose.json
python examples/sqd/replay.py --probe lazy --output /path/to/new-run/lazy.json
```

The output includes exact installed versions, input/seed, command, source SHA256,
assertion counts and numerical results. Capture stderr/stdout and exit status externally
(including failures, which do not produce a success JSON). Hash the final JSON externally.
`manifest.json` records the original read-only prototype hashes and persisted file hashes;
`validate_manifest()` requires exactly the five persisted files and fails on missing,
modified or escaping sources (including symlink escapes). Original sources
need not be installed to rerun this bundle. Do not edit the historical prototype files.

| Probe | Actual checks | Scope and adaptations |
| --- | ---: | --- |
| h2 | 14 | H2/STO-3G, (2e,2o), seed 20260721, 50000 shots, 2 UCJ repetitions; 6 physical/import checks plus 8 channel-mask/nonzero/rejection checks. Both interaction channels accept upper-triangular pairs including diagonal. |
| reference | 21 | Original part E and its historical fitted constants, with `math.comb` replacing `cistring.num_strings`; no new scientific measurements. |
| transpose | 6 | H2O (4e,4o), independent vacuum RHF/ao2mo/CASCI oracle, fixed-MO helper loaded from this checkout; inverse permutation (0,3,1,2) and non-degenerate missing-permutation negative witness. This second RHF is an independent test oracle, not the planned production assembly. |
| lazy | 24 | Six assertions in each of four fresh-interpreter install profiles; restores disposable package around copied helper. The original “six” harness recorded seven entries; this version groups two explicit-export outcomes into one assertion and checks unknown attributes directly. |

The reference replay deliberately preserves old behavior: default 120 seconds and
GB budgets, plugin before selected-CI, extrapolation text for uncertainty. These differ
from T01's default 900 seconds/8192 decimal MB, T0→T1→ordered T1+→T2, dual budgets and
unknown-uncertainty `None`. Passing historical replay does **not** validate the new
resolver. The helper's PySCF producer convention remains unchanged. Lazy export replay
uses a temporary package, not production q2m3: root's existing eager PennyLane debt is
outside this task, and production lazy exports are a later task.

No generated run evidence belongs in this directory. Bundle integrity tests run with:

```bash
python -m pytest -o addopts='' tests/sqd/test_replay_manifest.py
```

The lazy oracle derives exact expected exports independently from installed module
availability and each blocked-module list. Both `__all__` and star imports must match
that set; expected exports must resolve to callables. Missing exports must raise an
actionable ImportError. Lightweight copied-helper mutations verify that always-empty
exports and always-hinted errors are rejected, individually and together.

The SQD-only profile blocks `catalyst` but preserves `jax`: ffsim 0.0.84 itself
imports JAX, so removing both Catalyst and JAX while claiming SQD is installed is
an invalid dependency profile. The neither-extra profile may block JAX too. A
regression uses an ffsim stub that imports JAX, and executes all four profiles.
Failed child interpreters print their stderr and propagate a nonzero exit.
