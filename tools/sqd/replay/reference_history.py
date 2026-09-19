"""Historical reference model; not the production reference contract."""

from math import comb

T0_DETS_PER_S = 1.638e5  # fit to the harder point (16o); predicts 72 s at 14o vs 52 s measured
T0_VECS = 6.0  # Davidson working vectors, fit to 16o
T0_RSS_FLOOR_GB = 2.6  # integral-transform working set, fit to 14o
T1_DETS_PER_S = 4.37e6  # 1.013e9 full-space dets / 231.5 s (selected_ci, 18o)
T1_FULL_NDET_MAX = 2e9  # selected_ci direct: measured OK at 1.013e9 (norb=18)


def resolve_reference(norb, nelec, budget=None):
    """Deterministic reference-tier selection.  budget: wall_s / rss_gb / plugins."""
    b = budget or {}
    wall, rss = b.get("wall_s", 120.0), b.get("rss_gb", 8.0)
    plugins = set(b.get("plugins", [])) & {"dice", "block2"}
    na, nb = (nelec, nelec) if isinstance(nelec, int) else nelec
    ndet = int(comb(norb, na)) * int(comb(norb, nb))
    t0_wall = 0.2 + ndet / T0_DETS_PER_S
    t0_rss = max(T0_RSS_FLOOR_GB, 0.3 + 8.0 * ndet * T0_VECS / 1e9)
    if t0_wall <= wall and t0_rss <= rss:  # T0 always wins when affordable:
        return {
            "tier": "T0",
            "method": "exact_casci",
            "ndet": ndet,  # measured T1 is
            "uncertainty_mHa": 0.0,
            "downgrade_reason": None,  # 5.5-7.4x SLOWER
            "est_wall_s": t0_wall,
            "est_rss_gb": t0_rss,
        }  # than T0 at norb<=4
    why = "; ".join(
        [f"T0 est_wall {t0_wall:.0f}s>{wall:.0f}s"] * (t0_wall > wall)
        + [f"T0 est_rss {t0_rss:.1f}GB>{rss:.1f}GB"] * (t0_rss > rss)
    )
    if plugins:
        return {
            "tier": "T1+",
            "method": sorted(plugins)[0],
            "ndet": ndet,
            "uncertainty_mHa": None,
            "downgrade_reason": why,
        }
    if ndet <= T1_FULL_NDET_MAX:
        return {
            "tier": "T1",
            "method": "selected_ci_pyscf",
            "ndet": ndet,
            "uncertainty_mHa": "extrapolated from >=2 select_cutoff",
            "downgrade_reason": why,
            "est_wall_s": ndet / T1_DETS_PER_S,
        }
    return {
        "tier": "T2",
        "method": "ccsd_t",
        "ndet": ndet,
        "uncertainty_mHa": None,
        "downgrade_reason": why + f"; full_ndet {ndet:.2g} > T1 ceiling {T1_FULL_NDET_MAX:.2g}",
        "requires_t1_diagnostic_flag": True,
    }


def part_E():
    """Assert resolve_reference against every measured boundary."""
    checks = []

    def chk(desc, got, want):
        ok = got == want
        checks.append({"check": desc, "got": got, "want": want, "pass": ok})
        return ok

    # --- boundaries measured in parts A/B ---
    chk("B: AC-1 H2 (2e,2o) 0.012s", resolve_reference(2, (1, 1))["tier"], "T0")
    chk("B: AC-2 H3O+ (4e,4o) 0.0004s", resolve_reference(4, (2, 2))["tier"], "T0")
    chk("B: AC-3 (6e,6o) 0.018s", resolve_reference(6, (3, 3))["tier"], "T0")
    chk("C: N2 (10e,10o) 1.25s/0.28GB", resolve_reference(10, (5, 5))["tier"], "T0")
    chk(
        "A1/B: N2 (14e,14o) 52-76s/2.59GB, 120s+8GB budget",
        resolve_reference(14, (7, 7))["tier"],
        "T0",
    )
    chk(
        "A1: same case, 8GB->2GB budget -> RSS downgrade (measured 2.53GB)",
        resolve_reference(14, (7, 7), {"rss_gb": 2.0})["tier"],
        "T1",
    )
    chk(
        "A1: same case, 120s->30s budget -> wall downgrade (measured 52.1s)",
        resolve_reference(14, (7, 7), {"wall_s": 30.0})["tier"],
        "T1",
    )
    chk(
        "A1: (14e,16o) exact MEASURED 798.3s/6.23GB -> T1 under default 120s budget",
        resolve_reference(16, (7, 7))["tier"],
        "T1",
    )
    chk(
        "A1: (14e,16o) with 1200s/8GB budget -> T0 IS feasible (measured, not guessed)",
        resolve_reference(16, (7, 7), {"wall_s": 1200.0})["tier"],
        "T0",
    )
    chk(
        "A1: (14e,16o) 1200s but only 4GB -> RSS downgrade (measured 6.23GB)",
        resolve_reference(16, (7, 7), {"wall_s": 1200.0, "rss_gb": 4.0})["tier"],
        "T1",
    )
    chk(
        "model must be conservative: est_wall(16o) >= measured 798.3s",
        resolve_reference(16, (7, 7), {"wall_s": 1e9})["est_wall_s"] >= 798.3,
        True,
    )
    chk(
        "model must bracket measured RSS(14o)=2.53GB within 2x",
        0.5 <= resolve_reference(14, (7, 7), {"wall_s": 1e9})["est_rss_gb"] / 2.53 <= 2.0,
        True,
    )
    chk(
        "model must bracket measured RSS(16o)=6.23GB within 2x",
        0.5 <= resolve_reference(16, (7, 7), {"wall_s": 1e9})["est_rss_gb"] / 6.23 <= 2.0,
        True,
    )
    chk("A2: (14e,18o) ndet=1.01e9 -> T1", resolve_reference(18, (7, 7))["tier"], "T1")
    chk("(16e,16o) half-filled ndet=1.66e8 -> T1", resolve_reference(16, (8, 8))["tier"], "T1")
    chk(
        "(24e,24o) ndet=7.4e12 -> T2 (must carry T1-diagnostic flag)",
        resolve_reference(24, (12, 12))["tier"],
        "T2",
    )
    chk(
        "T2 result demands the D-part red flag",
        resolve_reference(24, (12, 12)).get("requires_t1_diagnostic_flag"),
        True,
    )
    chk(
        "plugin present at 16o -> T1+",
        resolve_reference(16, (8, 8), {"plugins": ["dice"]})["tier"],
        "T1+",
    )
    chk(
        "B anomaly rule: T0 affordable => never downgrade (T1 was 5.5-7.4x slower)",
        resolve_reference(2, (1, 1), {"plugins": ["dice"]})["tier"],
        "T0",
    )
    r = resolve_reference(16, (8, 8))
    checks.append(
        {
            "check": "downgrade_reason non-empty when not T0",
            "got": r["downgrade_reason"],
            "want": "<non-empty>",
            "pass": bool(r["downgrade_reason"]),
        }
    )
    r0 = resolve_reference(6, (3, 3))
    checks.append(
        {
            "check": "T0 uncertainty == 0.0",
            "got": r0["uncertainty_mHa"],
            "want": 0.0,
            "pass": r0["uncertainty_mHa"] == 0.0,
        }
    )
    return {
        "part": "E",
        "checks": checks,
        "n_pass": sum(c["pass"] for c in checks),
        "n_total": len(checks),
    }
