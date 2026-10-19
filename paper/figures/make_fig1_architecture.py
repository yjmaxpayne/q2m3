#!/usr/bin/env python3
"""Generate q2m3 JOSS Figure 1 (fig:architecture).

Renders a colourblind-safe (Okabe-Ito) SVG of the q2m3 software architecture:
two public entry points -- ``q2m3.core.estimate_resources`` (EFTQC resource
ledger via PennyLane DoubleFactorization) and ``q2m3.solvation.run_solvation``
(three-mode QPE Metropolis loop) -- over a shared PySCF / PennyLane / Catalyst
foundation. Structure verified against src/q2m3 (codegraph) and config.py:120.

Run:  uv run --with cairosvg python make_fig1_architecture.py
Out:  fig1_architecture.svg  +  fig1_architecture.png
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
SVG = HERE / "fig1_architecture.svg"
PNG = HERE / "fig1_architecture.png"

# --- Okabe-Ito colourblind-safe palette -------------------------------------
BLUE, ORANGE, GREEN = "#0072B2", "#E69F00", "#009E73"
SKY, VERM, PURPLE = "#56B4E9", "#D55E00", "#CC79A7"
GRAY, INK = "#6F6F6F", "#1F1F1F"
F_ORANGE, F_BLUE, F_GREEN, F_GRAY = "#FBF0DA", "#E3EEF7", "#DCF1EA", "#EFEFEF"

W, H = 1140, 700
L: list[str] = []


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, s, size=12.5, anchor="middle", color=INK, weight="normal", mono=False):
    fam = "DejaVu Sans Mono, monospace" if mono else "Inter, Helvetica, Arial, sans-serif"
    L.append(
        f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" '
        f'fill="{color}" font-weight="{weight}" font-family="{fam}">{esc(s)}</text>'
    )


def box(
    x, y, w, h, stroke, fill, title, sub=None, *, sw=1.8, t_size=13, s_size=9.6, mono_title=True
):
    L.append(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" ry="8" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
    )
    cx = x + w / 2
    if sub:
        text(cx, y + h / 2 - 3, title, size=t_size, color=INK, weight="bold", mono=mono_title)
        text(cx, y + h / 2 + 13, sub, size=s_size, color="#444444")
    else:
        text(cx, y + h / 2 + 4, title, size=t_size, color=INK, weight="bold", mono=mono_title)


def chip(x, y, w, h, stroke, fill, label, *, sw=1.8, size: float = 11):
    L.append(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" ry="6" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
    )
    text(x + w / 2, y + h / 2 + 4, label, size=size, color=INK, weight="bold", mono=True)


def group(x, y, w, h, stroke, label):
    L.append(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" ry="12" '
        f'fill="none" stroke="{stroke}" stroke-width="1.3" stroke-dasharray="6 4" opacity="0.8"/>'
    )
    text(x + 12, y + 18, label, size=11.5, anchor="start", color=stroke, weight="bold")


def vflow(x, y1, y2, color=BLUE):
    L.append(
        f'<line x1="{x}" y1="{y1}" x2="{x}" y2="{y2}" stroke="{color}" '
        f'stroke-width="2.2" marker-end="url(#flow)"/>'
    )


# --- canvas / defs ----------------------------------------------------------
L.append(
    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Inter, Helvetica, Arial, sans-serif">'
)
L.append("<defs>")
L.append(
    '<marker id="flow" markerWidth="10" markerHeight="8" refX="8" refY="4" orient="auto">'
    f'<path d="M0,0 L9,4 L0,8 z" fill="{BLUE}"/></marker>'
)
L.append(
    '<marker id="loop" markerWidth="10" markerHeight="8" refX="8" refY="4" orient="auto">'
    f'<path d="M0,0 L9,4 L0,8 z" fill="{PURPLE}"/></marker>'
)
L.append("</defs>")
L.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#FFFFFF"/>')

# --- title ------------------------------------------------------------------
text(W / 2, 32, "q2m3 software architecture", size=18, weight="bold")
text(
    W / 2,
    52,
    "two public entry points over a shared PySCF · PennyLane · Catalyst foundation",
    size=11.5,
    color="#555555",
)

# --- entry points -----------------------------------------------------------
box(
    120,
    76,
    360,
    46,
    BLUE,
    "#FFFFFF",
    "estimate_resources()",
    "q2m3.core — EFTQC resource path",
    sw=2.4,
)
box(
    660,
    76,
    360,
    46,
    BLUE,
    "#FFFFFF",
    "run_solvation(config)",
    "q2m3.solvation — MC solvation path",
    sw=2.4,
)

# --- legend (centre gap) ----------------------------------------------------
L.append(
    '<rect x="498" y="78" width="144" height="44" rx="6" fill="#FAFAFA" stroke="#DDDDDD" stroke-width="1"/>'
)
L.append(
    f'<line x1="510" y1="92" x2="540" y2="92" stroke="{BLUE}" stroke-width="2.2" marker-end="url(#flow)"/>'
)
text(548, 96, "data / control flow", size=9.5, anchor="start", color="#444444")
L.append(
    f'<line x1="510" y1="110" x2="540" y2="110" stroke="{PURPLE}" stroke-width="2.2" marker-end="url(#loop)"/>'
)
text(548, 114, "MC iteration", size=9.5, anchor="start", color="#444444")

# --- resource path (left column) --------------------------------------------
group(124, 150, 352, 264, ORANGE, "EFTQC resource estimation")
box(
    140,
    168,
    320,
    48,
    ORANGE,
    F_ORANGE,
    "interfaces",
    "PySCF→PennyLane · fixed-MO MM embedding",
    mono_title=True,
)
box(
    140,
    256,
    320,
    48,
    ORANGE,
    F_ORANGE,
    "DoubleFactorization",
    "PennyLane DF · active space",
    mono_title=True,
)
box(
    140,
    344,
    320,
    52,
    ORANGE,
    F_ORANGE,
    "EFTQCResources",
    "λ (1-norm) · logical qubits · Toffoli",
    mono_title=True,
)
vflow(300, 122, 168)
vflow(300, 216, 256)
vflow(300, 304, 344)

# --- solvation path (right column) ------------------------------------------
group(664, 150, 352, 388, BLUE, "three-mode QPE solvation")
box(
    680,
    168,
    320,
    44,
    BLUE,
    F_BLUE,
    "Config",
    "hamiltonian_mode · n_mc_steps · n_est",
    mono_title=True,
)
# mode selector chips
chip(680, 228, 100, 32, GRAY, F_GRAY, "hf_corrected", size=10)
chip(788, 228, 86, 32, BLUE, F_BLUE, "fixed", size=10.5)
chip(882, 228, 118, 32, GREEN, F_GREEN, "dynamic", sw=2.4, size=10.5)
text(
    840,
    276,
    'hamiltonian_mode: Literal["hf_corrected","fixed","dynamic"]',
    size=8.6,
    color="#777777",
)
box(
    680,
    292,
    320,
    48,
    BLUE,
    F_BLUE,
    "Circuit Builder",
    "QPE @qjit · TrotterProduct · +LLVM-IR cache",
    mono_title=True,
)
box(680, 380, 320, 48, BLUE, F_BLUE, "Energy", "E_QPE+E_MM  |  E_HF+δ_corr-pol", mono_title=True)
box(
    680, 468, 320, 48, GREEN, F_GREEN, "MC Loop — Metropolis", "accept / reject ΔE", mono_title=True
)
vflow(840, 122, 168)
# Config -> mode strip -> Circuit Builder
L.append(f'<line x1="840" y1="212" x2="840" y2="222" stroke="{BLUE}" stroke-width="1.6"/>')
L.append(f'<line x1="730" y1="222" x2="941" y2="222" stroke="{BLUE}" stroke-width="1.6"/>')
for mx in (730, 831, 941):
    L.append(f'<line x1="{mx}" y1="222" x2="{mx}" y2="228" stroke="{BLUE}" stroke-width="1.6"/>')
vflow(840, 260, 292)
vflow(840, 340, 380)
vflow(840, 428, 468)
# MC loop feedback (purple, orthogonal, right margin)
L.append(
    f'<path d="M1000,492 H1052 V190 H1000" fill="none" stroke="{PURPLE}" '
    f'stroke-width="2.2" marker-end="url(#loop)"/>'
)
L.append('<rect x="1010" y="332" width="76" height="16" fill="#FFFFFF" opacity="0.92"/>')
text(1048, 344, "next MC step", size=9, color=PURPLE, weight="bold")

# --- foundation band --------------------------------------------------------
text(
    60,
    590,
    "Foundation libraries (builds on)",
    size=11,
    anchor="start",
    color="#666666",
    weight="bold",
)
L.append(
    f'<rect x="60" y="600" width="360" height="56" rx="7" fill="{F_ORANGE}" stroke="{ORANGE}" stroke-width="1.6"/>'
)
L.append(f'<rect x="60" y="600" width="360" height="6" rx="3" fill="{ORANGE}"/>')
text(240, 626, "PySCF", size=13, weight="bold")
text(240, 644, "RHF · integrals · qmmm.mm_charge", size=9.4, color="#444444")
L.append(
    f'<rect x="440" y="600" width="320" height="56" rx="7" fill="{F_BLUE}" stroke="{BLUE}" stroke-width="1.6"/>'
)
L.append(f'<rect x="440" y="600" width="320" height="6" rx="3" fill="{BLUE}"/>')
text(600, 626, "PennyLane", size=13, weight="bold")
text(600, 644, "qchem · DoubleFactorization · QPE", size=9.4, color="#444444")
L.append(
    f'<rect x="780" y="600" width="300" height="56" rx="7" fill="{F_GREEN}" stroke="{GREEN}" stroke-width="1.6"/>'
)
L.append(f'<rect x="780" y="600" width="300" height="6" rx="3" fill="{GREEN}"/>')
text(930, 626, "Catalyst", size=13, weight="bold")
text(930, 644, "@qjit · MLIR→LLVM · IR cache", size=9.4, color="#444444")

L.append("</svg>")

SVG.write_text("\n".join(L), encoding="utf-8")
ET.parse(SVG)  # raises on malformed XML
print(f"[ok] wrote {SVG.name} ({SVG.stat().st_size} B), XML valid")

import cairosvg  # noqa: E402

cairosvg.svg2png(url=str(SVG), write_to=str(PNG), scale=2)
print(f"[ok] wrote {PNG.name} ({PNG.stat().st_size} B)")
