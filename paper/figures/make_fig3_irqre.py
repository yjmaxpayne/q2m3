#!/usr/bin/env python3
"""Generate q2m3 JOSS Figure 3 (fig:irqre): IR<->QRE compile-cost scaling.

Peak compile-time RSS vs LLVM-IR operation count for the ``fixed`` and
``dynamic`` (runtime-coefficient) QPE circuit families. Both families follow a
linear law; the ``dynamic`` slope is ~29x steeper than ``fixed``. The H3O+
target is *measured* in ``fixed`` mode (fits in ~5.7 GB) but only *projected*
in ``dynamic`` mode (~100 GB, an extrapolation past the single-node memory
ceiling) -- the quantitative reason ``dynamic`` H3O+ is infeasible while
``fixed`` succeeds.

Data: paper/figures/data/irqre_fit_diagnostics.json (q2m3 S3-01 artifact,
fixed R^2=0.94 / dynamic R^2=0.98; values reproduced here by OLS).

Run:  uv run python make_fig3_irqre.py     (project .venv: numpy/scipy/matplotlib)
Out:  fig3_irqre_compile_scaling.png + .pdf
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "irqre_fit_diagnostics.json"
PNG = HERE / "fig3_irqre_compile_scaling.png"
PDF = HERE / "fig3_irqre_compile_scaling.pdf"

FIXED, DYN = "#E69F00", "#0072B2"  # Okabe-Ito orange / blue (CB-safe)
SAFETY_GB = 0.85 * 64.0  # 54.4 GB safety boundary on a 64 GB node

with DATA.open() as fh:
    d = json.load(fh)


def ols_with_band(rows, x_grid):
    """Return slope, intercept, R^2, and the 95% CI band over x_grid."""
    x = np.array([r["ir_ops_lower_bound"] for r in rows], float)
    y = np.array([r["compile_rss_gb"] for r in rows], float)
    n = len(x)
    b, a = np.polyfit(x, y, 1)  # y = a + b x
    yhat = a + b * x
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    s = np.sqrt(ss_res / (n - 2))
    sxx = float(np.sum((x - x.mean()) ** 2))
    tval = stats.t.ppf(0.975, n - 2)
    se = s * np.sqrt(1.0 / n + (x_grid - x.mean()) ** 2 / sxx)
    band = tval * se
    return dict(x=x, y=y, a=a, b=b, r2=r2, line=a + b * x_grid, band=band)


fx_rows = d["fixed"]["rows"]
dy_rows = d["dynamic_measured_only"]["rows"]
dy_xmax = d["dynamic_measured_only"]["x_max"]  # 2100 (measured range)
h3o_dyn_x = d["target_x_ir_ops"]["dynamic"]  # 7220 (projection target)
h3o_dyn_y = d["dynamic_measured_only"]["best"]["target_y_gb"]
h3o_dyn_lo = d["dynamic_measured_only"]["best"]["target_ci_low_gb"]
h3o_dyn_hi = d["dynamic_measured_only"]["best"]["target_ci_high_gb"]

xg = np.linspace(0, 8000, 400)
fx = ols_with_band(fx_rows, xg)
dy = ols_with_band(dy_rows, xg)

fig, ax = plt.subplots(figsize=(8.0, 5.2), dpi=200)

# memory-safety boundary
ax.axhline(SAFETY_GB, color="#888888", lw=1.2, ls=(0, (6, 4)))
ax.text(
    7950,
    SAFETY_GB + 1.5,
    "memory-safety boundary (0.85 × 64 GB)",
    ha="right",
    va="bottom",
    fontsize=8.5,
    color="#666666",
)

# --- fixed family --------------------------------------------------------
ax.fill_between(xg, fx["line"] - fx["band"], fx["line"] + fx["band"], color=FIXED, alpha=0.15, lw=0)
ax.plot(xg, fx["line"], color=FIXED, lw=2.0)
ax.scatter(
    fx["x"],
    fx["y"],
    s=70,
    color=FIXED,
    edgecolor="white",
    lw=1.0,
    zorder=5,
    label=f"fixed  (R²={fx['r2']:.2f})",
)

# --- dynamic family: solid over measured range, dashed extrapolation -----
m = xg <= dy_xmax
dy_line, dy_band = np.asarray(dy["line"]), np.asarray(dy["band"])
ax.fill_between(xg, dy_line - dy_band, dy_line + dy_band, color=DYN, alpha=0.13, lw=0)
ax.plot(xg[m], dy_line[m], color=DYN, lw=2.0)
ax.plot(xg[~m], dy_line[~m], color=DYN, lw=2.0, ls=(0, (5, 3)))
ax.scatter(
    dy["x"],
    dy["y"],
    s=70,
    marker="s",
    color=DYN,
    edgecolor="white",
    lw=1.0,
    zorder=5,
    label=f"dynamic (runtime-coeff.)  (R²={dy['r2']:.2f})",
)

# --- H3O+ highlights -----------------------------------------------------
# fixed H3O+ is the largest measured fixed point (x = 7720)
h3o_fx = next(r for r in fx_rows if "H3O" in r["label"])
ax.scatter(
    [h3o_fx["ir_ops_lower_bound"]],
    [h3o_fx["compile_rss_gb"]],
    s=150,
    facecolor="none",
    edgecolor=FIXED,
    lw=2.0,
    zorder=6,
)
ax.annotate(
    "H₃O⁺ fixed (measured, 5.7 GB)",
    (h3o_fx["ir_ops_lower_bound"], h3o_fx["compile_rss_gb"]),
    xytext=(-12, 18),
    textcoords="offset points",
    fontsize=8.5,
    color=FIXED,
    ha="right",
    weight="bold",
)
# dynamic H3O+ is a projection (extrapolation) with CI
ax.errorbar(
    [h3o_dyn_x],
    [h3o_dyn_y],
    yerr=[[h3o_dyn_y - h3o_dyn_lo], [h3o_dyn_hi - h3o_dyn_y]],
    fmt="*",
    ms=18,
    color=DYN,
    ecolor=DYN,
    elinewidth=1.5,
    capsize=5,
    zorder=6,
    markeredgecolor="white",
)
ax.annotate(
    f"H₃O⁺ dynamic\n(projected ≈{h3o_dyn_y:.0f} GB)",
    (h3o_dyn_x, h3o_dyn_y),
    xytext=(-14, -6),
    textcoords="offset points",
    fontsize=8.5,
    color=DYN,
    ha="right",
    va="center",
    weight="bold",
)

# --- fit equations + slope ratio -----------------------------------------
ax.text(
    0.035,
    0.95,
    f"fixed:    RSS = {fx['a']:.2f} + {fx['b']:.1e}·n_IR\n"
    f"dynamic:  RSS = {dy['a']:.2f} + {dy['b']:.1e}·n_IR\n"
    f"slope ratio ≈ {dy['b'] / fx['b']:.0f}× steeper",
    transform=ax.transAxes,
    fontsize=9.5,
    va="top",
    family="DejaVu Sans Mono",
    bbox=dict(boxstyle="round,pad=0.4", fc="#FAFAFA", ec="#DDDDDD"),
)

ax.set_xlabel("LLVM-IR operation count  (n_est × Trotter × Pauli terms)", fontsize=11.5)
ax.set_ylabel("peak compile-time RSS (GB)", fontsize=11.5)
ax.set_title(
    "Compile-cost scaling of the fixed and dynamic QPE circuit families",
    fontsize=12.5,
    weight="bold",
)
ax.set_xlim(0, 8000)
ax.set_ylim(0, 130)
ax.grid(True, color="#E6E6E6", lw=0.8)
ax.set_axisbelow(True)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.legend(loc="center left", bbox_to_anchor=(0.035, 0.62), frameon=True, fontsize=9.5)

fig.tight_layout()
fig.savefig(PNG, bbox_inches="tight")
fig.savefig(PDF, bbox_inches="tight")
print(f"[ok] fixed: slope={fx['b']:.3e} intercept={fx['a']:.3f} R2={fx['r2']:.4f}")
print(f"[ok] dynamic: slope={dy['b']:.3e} intercept={dy['a']:.3f} R2={dy['r2']:.4f}")
print(f"[ok] dynamic H3O+ projection={h3o_dyn_y:.1f} GB CI=[{h3o_dyn_lo:.1f},{h3o_dyn_hi:.1f}]")
print(f"[ok] wrote {PNG.name} + {PDF.name}")
