#!/usr/bin/env python3
"""Generate q2m3 JOSS Figure 2 (fig:dynamic): the three QPE solvation modes.

For each mode (``fixed`` / ``hf_corrected`` / ``dynamic``) the lane shows what
is compiled once, what runs per Metropolis MC step, the acceptance energy, and
whether the mode captures the correlation-polarisation coupling delta_corr-pol.
The ``dynamic`` lane is the distinctive contribution: a parametrised QPE circuit
is compiled once (runtime coefficients, ``check_hermitian=False``) and reused
across every solvent configuration -- no per-configuration recompilation.

Timings: H2 + 10 TIP3P, 500 MC steps (dev/wiki/solvation.md, verified SSOT).

Run:  uv run --with cairosvg python make_fig2_three_modes.py
Out:  fig2_three_modes.svg  +  fig2_three_modes.png
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
SVG = HERE / "fig2_three_modes.svg"
PNG = HERE / "fig2_three_modes.png"

# Okabe-Ito colourblind-safe palette
BLUE, ORANGE, GREEN = "#0072B2", "#E69F00", "#009E73"
PURPLE, INK = "#CC79A7", "#1F1F1F"
F_BLUE, F_ORANGE, F_GREEN = "#E3EEF7", "#FBF0DA", "#DCF1EA"

W, H = 1180, 624
L: list[str] = []


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, s, size=12.0, anchor="middle", color=INK, weight="normal", mono=False, italic=False):
    fam = "DejaVu Sans Mono, monospace" if mono else "Inter, Helvetica, Arial, sans-serif"
    style = ' font-style="italic"' if italic else ""
    L.append(
        f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" '
        f'fill="{color}" font-weight="{weight}" font-family="{fam}"{style}>{esc(s)}</text>'
    )


def panel(x, y, w, h, stroke, fill, sw=1.7):
    L.append(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" ry="8" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
    )


def stack(cx, top, items, gap=15):
    y = top
    for it in items:
        s, size, weight, color, mono = (it + (False,))[:5] if len(it) == 4 else it
        text(cx, y, s, size=size, weight=weight, color=color, mono=mono)
        y += gap


def flow(x1, y1, x2, y2, color=BLUE, dash=None, marker="flow", sw=2.2):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    L.append(
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
        f'stroke-width="{sw}"{d} marker-end="url(#{marker})"/>'
    )


# station x-centres / box geometry
CX_COMPILE, CX_STEP, CX_ACCEPT = 410, 710, 1014
BX_C, BX_S, BX_A = 282, 582, 880
BW_C, BW_S, BW_A = 256, 256, 268

L.append(
    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Inter, Helvetica, Arial, sans-serif">'
)
L.append("<defs>")
L.append(
    f'<marker id="flow" markerWidth="10" markerHeight="8" refX="8" refY="4" orient="auto"><path d="M0,0 L9,4 L0,8 z" fill="{BLUE}"/></marker>'
)
L.append(
    '<marker id="reuse" markerWidth="10" markerHeight="8" refX="8" refY="4" orient="auto"><path d="M0,0 L9,4 L0,8 z" fill="#888888"/></marker>'
)
L.append("</defs>")
L.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#FFFFFF"/>')

# title + subtitle
text(W / 2, 30, "Three QPE solvation modes", size=18, weight="bold")
text(
    W / 2,
    50,
    "what is compiled once, what runs per Metropolis MC step, and QPE's role  (H2 + 10 TIP3P, 500 MC steps)",
    size=11,
    color="#555555",
)

# column headers
for cx, lab in (
    (140, "mode"),
    (CX_COMPILE, "compile · one-time"),
    (CX_STEP, "per MC step"),
    (CX_ACCEPT, "Metropolis acceptance"),
):
    text(cx, 78, lab, size=11.5, weight="bold", color="#666666")

# ---- lane data -------------------------------------------------------------
# (cy, color, fill, mode, role, compile_lines, c_badge, step_lines, s_badge, accept_expr, tag)
lanes = [
    (
        128,
        BLUE,
        F_BLUE,
        "fixed",
        "vacuum H · QPE every step",
        [
            ("compile vacuum H once", 11, "bold", INK),
            ("@qjit zero-arg QPE circuit", 9.4, "normal", "#444"),
        ],
        "7.5 s",
        [("re-run compiled", 11, "bold", INK), ("QPE(H_vac)", 10, "normal", "#444", True)],
        "11.2 ms/step",
        "E_QPE(H_vac) + E_MM",
        "delta_corr-pol: NO  ·  sigma_QPE = 0 (MM-decoupled)  ·  fastest, total 15.5 s",
    ),
    (
        242,
        ORANGE,
        F_ORANGE,
        "hf_corrected",
        "classical accept · QPE diagnostic",
        [("@qjit MC loop", 11, "bold", INK), ("(QPE inlined in loop IR)", 9.4, "normal", "#444")],
        "224 s",
        [
            ("QMMM-RHF energy (PySCF)", 10, "bold", INK),
            ("QPE every N steps (24.5 ms)", 9.2, "normal", "#444"),
        ],
        "—",
        "E_HF(solvated) + E_MM",
        "delta_corr-pol: partial  ·  QPE used only as a diagnostic  ·  medium, total 233.6 s",
    ),
]

# common lanes (fixed, hf_corrected)
for cy, color, fill, mode, role, c_lines, c_badge, s_lines, s_badge, acc, tag in lanes:
    band_top = cy - 56
    L.append(
        f'<rect x="24" y="{band_top}" width="{W - 48}" height="100" rx="10" fill="{fill}" opacity="0.35"/>'
    )
    # mode chip + role
    panel(40, cy - 22, 200, 44, color, fill)
    text(140, cy + 4, mode, size=13, weight="bold", color=INK, mono=True)
    text(140, cy + 38, role, size=9.2, color="#555")
    # compile / step / accept panels
    panel(BX_C, cy - 36, BW_C, 72, color, "#FFFFFF")
    stack(CX_COMPILE, cy - 14, c_lines)
    text(CX_COMPILE, cy + 26, c_badge, size=12, weight="bold", color=color)
    panel(BX_S, cy - 36, BW_S, 72, color, "#FFFFFF")
    stack(CX_STEP, cy - 14, s_lines)
    if s_badge != "—":
        text(CX_STEP, cy + 26, s_badge, size=12, weight="bold", color=color)
    panel(BX_A, cy - 36, BW_A, 72, color, "#FFFFFF")
    text(CX_ACCEPT, cy - 6, "E =", size=10, color="#666")
    text(CX_ACCEPT, cy + 14, acc, size=11.5, weight="bold", color=INK, mono=True)
    # arrows
    flow(BX_C + BW_C, cy, BX_S - 2, cy, color="#888888", dash="5 4", marker="reuse", sw=1.8)
    flow(BX_S + BW_S, cy, BX_A - 2, cy, color=color)
    text((BX_C + BW_C + BX_S) / 2, cy - 8, "reused", size=8.6, color="#888")
    # tag below band
    text(W / 2, band_top + 116, tag, size=9.6, color="#555", italic=True)

# ---- dynamic lane (taller, two-phase, highlighted) -------------------------
dy = 446
band_top = dy - 70
L.append(
    f'<rect x="24" y="{band_top}" width="{W - 48}" height="132" rx="10" fill="{F_GREEN}" opacity="0.5"/>'
)
L.append(
    f'<rect x="24" y="{band_top}" width="{W - 48}" height="132" rx="10" fill="none" stroke="{GREEN}" stroke-width="2" stroke-dasharray="7 4"/>'
)
text(
    40,
    band_top + 16,
    "compile once / run many — the distinctive mode",
    size=10,
    anchor="start",
    color=GREEN,
    weight="bold",
)
# mode chip
panel(40, dy - 18, 200, 50, GREEN, F_GREEN, sw=2.4)
text(140, dy + 6, "dynamic", size=14, weight="bold", color=INK, mono=True)
text(140, dy + 44, "MM-embedded H · QPE every step", size=9.2, color="#555")
# Phase 1 compile panel
panel(BX_C, dy - 40, BW_C, 84, GREEN, "#FFFFFF", sw=2.0)
stack(
    CX_COMPILE,
    dy - 22,
    [
        ("build H_eff template →", 10.2, "bold", INK),
        ("@qjit precompile parametrised QPE", 9.0, "normal", "#444"),
        ("(runtime coeffs, check_hermitian=False)", 8.4, "normal", "#444"),
        ("→ compiled closure", 9.4, "normal", "#444"),
    ],
    gap=13,
)
text(CX_COMPILE, dy + 38, "220 s · one-time", size=11, weight="bold", color=GREEN)
# Phase 2 per-step panel
panel(BX_S, dy - 40, BW_S, 84, GREEN, "#FFFFFF", sw=2.0)
stack(
    CX_STEP,
    dy - 18,
    [
        ("patch JAX coeffs from", 10, "bold", INK),
        ("Δh_MM (MO basis)", 10, "normal", "#444", True),
        ("→ reuse compiled closure", 9.4, "normal", "#444"),
    ],
    gap=14,
)
text(CX_STEP, dy + 38, "5.5 ms/step · 494× no-recompile", size=10, weight="bold", color=GREEN)
# accept panel
panel(BX_A, dy - 40, BW_A, 84, GREEN, "#FFFFFF", sw=2.0)
text(CX_ACCEPT, dy - 14, "E =", size=10, color="#666")
text(CX_ACCEPT, dy + 6, "E_QPE(H_eff(R_MM)) + E_MM", size=11, weight="bold", color=INK, mono=True)
text(CX_ACCEPT, dy + 30, "delta_corr-pol: YES (full)", size=9.6, weight="bold", color=GREEN)
# arrows
flow(BX_C + BW_C, dy, BX_S - 2, dy, color="#888888", dash="5 4", marker="reuse", sw=1.8)
flow(BX_S + BW_S, dy, BX_A - 2, dy, color=GREEN)
text((BX_C + BW_C + BX_S) / 2, dy - 8, "reused", size=8.6, color="#888")

# ---- bottom callout --------------------------------------------------------
cyb = 596
L.append(
    f'<rect x="24" y="{cyb - 22}" width="{W - 48}" height="34" rx="8" fill="#FAFAFA" stroke="#DDDDDD" stroke-width="1"/>'
)
text(
    W / 2,
    cyb,
    "dynamic mode reassembles the Hamiltonian as qml.dot(coeffs, ops) with JAX-traced coefficients inside one @qjit circuit — reused across all solvent configurations, so QPE per MC step is tractable.",
    size=9.8,
    color="#444",
)

L.append("</svg>")

SVG.write_text("\n".join(L), encoding="utf-8")
ET.parse(SVG)
print(f"[ok] wrote {SVG.name} ({SVG.stat().st_size} B), XML valid")

import cairosvg  # noqa: E402

cairosvg.svg2png(url=str(SVG), write_to=str(PNG), scale=2)
print(f"[ok] wrote {PNG.name} ({PNG.stat().st_size} B)")
