#!/usr/bin/env python3
# Copyright (c) 2025 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT
"""Generate manuscript Figure 3: representative solvent-shell snapshots."""

from __future__ import annotations

import argparse
import glob
import shutil
import subprocess
from pathlib import Path

import matplotlib
import numpy as np
from ase import Atoms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "output" / "h2_mc_structure_analysis"
DEFAULT_OUTPUT = DEFAULT_INPUT_DIR / "figure3_solvent_shell_snapshots.pdf"

ELEMENT_COLORS = {
    "H": "#E8E8E8",
    "O": "#D55E00",
    "N": "#0072B2",
    "C": "#222222",
}
BALL_RADII = {"H": 0.18, "O": 0.28, "N": 0.27, "C": 0.26}
MAIN_VIEW_AZIMUTH_DEG = 45.0
MAIN_VIEW_ELEVATION_DEG = 25.0
PERSPECTIVE_FOCAL_LENGTH = 16.0


def parse_xyz(path: Path) -> tuple[list[str], np.ndarray, str]:
    """Parse one XYZ file."""
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        raise ValueError(f"{path} is not a valid XYZ file")
    n_atoms = int(lines[0])
    comment = lines[1]
    atom_lines = lines[2 : 2 + n_atoms]
    if len(atom_lines) != n_atoms:
        raise ValueError(f"{path} declares {n_atoms} atoms but has {len(atom_lines)} lines")
    symbols: list[str] = []
    coords: list[list[float]] = []
    for line in atom_lines:
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"bad XYZ atom line in {path}: {line}")
        symbols.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return symbols, np.array(coords, dtype=float), comment


def _configure_style() -> None:
    """Configure publication-friendly matplotlib defaults."""
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _atom_colors(symbols: list[str]) -> list[str]:
    """Return atom colors, highlighting the H2 solute in blue."""
    colors: list[str] = []
    for index, symbol in enumerate(symbols):
        if index < 2 and symbols[:2] == ["H", "H"]:
            colors.append("#0072B2")
        else:
            colors.append(ELEMENT_COLORS.get(symbol, "#999999"))
    return colors


def _atom_radii(symbols: list[str]) -> list[float]:
    """Return compact ball radii for projected rendering."""
    return [BALL_RADII.get(symbol, 0.22) for symbol in symbols]


def _atoms_from_xyz(xyz_path: Path) -> tuple[Atoms, str]:
    """Return an ASE Atoms object and XYZ comment."""
    symbols, coords, comment = parse_xyz(xyz_path)
    return Atoms(symbols=symbols, positions=coords), comment


def _bond_pairs(atoms: Atoms) -> list[tuple[int, int]]:
    """Return explicit H2 and TIP3P intramolecular bond pairs."""
    symbols = atoms.get_chemical_symbols()
    pairs: list[tuple[int, int]] = []
    if len(symbols) >= 2 and symbols[:2] == ["H", "H"]:
        pairs.append((0, 1))
    for start in range(2, len(symbols), 3):
        if start + 2 < len(symbols) and symbols[start : start + 3] == ["O", "H", "H"]:
            pairs.extend([(start, start + 1), (start, start + 2)])
    return pairs


def _rotation_matrix(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """Return a 3D rotation matrix for the main perspective view."""
    azimuth = np.deg2rad(azimuth_deg)
    elevation = np.deg2rad(elevation_deg)
    rot_z = np.array(
        [
            [np.cos(azimuth), -np.sin(azimuth), 0.0],
            [np.sin(azimuth), np.cos(azimuth), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rot_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(elevation), -np.sin(elevation)],
            [0.0, np.sin(elevation), np.cos(elevation)],
        ]
    )
    return rot_x @ rot_z


def _project_main_view(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return 2D projected coordinates plus positive depth values."""
    centered = coords - coords.mean(axis=0, keepdims=True)
    rotated = centered @ _rotation_matrix(MAIN_VIEW_AZIMUTH_DEG, MAIN_VIEW_ELEVATION_DEG).T
    depth = rotated[:, 1]
    depth_shift = depth - depth.min() + 2.0
    scale = PERSPECTIVE_FOCAL_LENGTH / (PERSPECTIVE_FOCAL_LENGTH + depth_shift)
    projected = np.column_stack((rotated[:, 0] * scale, rotated[:, 2] * scale))
    return projected, depth_shift


def _plot_main_view(ax, atoms: Atoms, comment: str, panel_label: str, step_label: str) -> None:
    """Plot a bond-plus-ball main perspective view."""
    coords = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    projected, depth = _project_main_view(coords)
    colors = _atom_colors(symbols)
    radii = _atom_radii(symbols)
    order = np.argsort(depth)[::-1]

    for i, j in _bond_pairs(atoms):
        mid_scale = 0.5 * (BALL_RADII.get(symbols[i], 0.22) + BALL_RADII.get(symbols[j], 0.22))
        ax.plot(
            [projected[i, 0], projected[j, 0]],
            [projected[i, 1], projected[j, 1]],
            color="#7A7A7A",
            lw=3.5 * mid_scale,
            solid_capstyle="round",
            zorder=1,
        )

    import matplotlib.patches as patches

    for index in order:
        radius = radii[index] * (0.92 + 0.08 * depth[index])
        circle = patches.Circle(
            (projected[index, 0], projected[index, 1]),
            radius=radius,
            facecolor=colors[index],
            edgecolor="#1F1F1F",
            linewidth=0.55,
            zorder=2 + float(depth[index]),
        )
        ax.add_patch(circle)

    ax.set_aspect("equal")
    ax.axis("off")
    ax.margins(0.22)
    ax.set_title(f"{panel_label}  {step_label}", loc="left")
    ax.text(0.02, 0.03, comment[:78], transform=ax.transAxes, fontsize=6.3, color="#555555")


def _projection_points(atoms: Atoms) -> tuple[np.ndarray, np.ndarray]:
    """Return water-oxygen positions and the H2 centroid."""
    coords = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    oxygen_positions = coords[[i for i, symbol in enumerate(symbols) if symbol == "O"]]
    h2_centroid = np.mean(coords[:2], axis=0)
    return oxygen_positions, h2_centroid


def _plot_projection(ax, atoms: Atoms, plane: str) -> None:
    """Plot water-oxygen and H2-centroid projections onto a Cartesian plane."""
    oxygen_positions, h2_centroid = _projection_points(atoms)
    axis_map = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
    i, j = axis_map[plane]
    if len(oxygen_positions) > 0:
        ax.scatter(
            oxygen_positions[:, i],
            oxygen_positions[:, j],
            s=36,
            c="#D55E00",
            edgecolors="#7A2B00",
            linewidths=0.4,
        )
    ax.scatter(
        [h2_centroid[i]],
        [h2_centroid[j]],
        s=62,
        c="#0072B2",
        marker="X",
        edgecolors="#003B5C",
        linewidths=0.45,
    )
    ax.set_aspect("equal")
    ax.grid(True, color="#E1E1E1", lw=0.6)
    ax.set_xlabel(f"{plane[0]} (A)")
    ax.set_ylabel(f"{plane[1]} (A)")
    ax.set_title(plane)
    ax.margins(0.12)


def _plot_projection_legend(ax) -> None:
    """Place the shared projection legend under the xz/yz panels."""
    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [],
            [],
            linestyle="None",
            marker="o",
            markersize=5.8,
            markerfacecolor="#D55E00",
            markeredgecolor="#7A2B00",
            markeredgewidth=0.6,
            label="water O",
        ),
        Line2D(
            [],
            [],
            linestyle="None",
            marker="X",
            markersize=7.2,
            markerfacecolor="#0072B2",
            markeredgecolor="#003B5C",
            markeredgewidth=0.6,
            label="H2 centroid",
        ),
    ]
    ax.axis("off")
    ax.legend(
        handles=handles,
        labels=[handle.get_label() for handle in handles],
        frameon=False,
        loc="center",
        ncol=2,
        fontsize=6.3,
        handletextpad=0.45,
        columnspacing=1.25,
    )


def _render_with_pymol(xyz_paths: list[Path], output: Path) -> bool:
    """Try PyMOL PNG rendering and compose a PDF; return success."""
    pymol = shutil.which("pymol")
    if pymol is None:
        return False
    render_dir = output.parent / "_figure3_pymol"
    render_dir.mkdir(parents=True, exist_ok=True)
    png_paths: list[Path] = []
    try:
        for xyz_path in xyz_paths:
            png_path = render_dir / f"{xyz_path.stem}.png"
            pml_path = render_dir / f"{xyz_path.stem}.pml"
            pml_path.write_text(
                "\n".join(
                    [
                        f"load {xyz_path}, mol",
                        "hide everything, mol",
                        "show spheres, mol",
                        "set sphere_scale, 0.26, mol",
                        "set bg_rgb, white",
                        "orient mol",
                        "ray 1200, 900",
                        f"png {png_path}, dpi=300",
                        "quit",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            subprocess.run(
                [pymol, "-cq", str(pml_path)],
                check=True,
                capture_output=True,
                text=True,
            )
            png_paths.append(png_path)
    except Exception:
        return False

    _configure_style()
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(png_paths), figsize=(3.0 * len(png_paths), 3.0))
    if len(png_paths) == 1:
        axes = [axes]
    for label, ax, png_path in zip("ABC", axes, png_paths, strict=False):
        ax.imshow(mpimg.imread(png_path))
        ax.set_title(f"{label}  {png_path.stem}", loc="left")
        ax.axis("off")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return True


def _render_with_ase(xyz_paths: list[Path], output: Path) -> Path:
    """Render bond-plus-ball snapshots and planar projections from ASE atoms."""
    _configure_style()
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(9.2, 3.05 * len(xyz_paths)))
    outer_gs = fig.add_gridspec(len(xyz_paths), 1, hspace=0.42)
    for row_index, xyz_path in enumerate(xyz_paths):
        row_gs = outer_gs[row_index].subgridspec(
            2,
            4,
            width_ratios=[1.75, 1.0, 1.0, 1.0],
            height_ratios=[1.0, 0.15],
            hspace=0.05,
            wspace=0.28,
        )
        atoms, comment = _atoms_from_xyz(xyz_path)
        step_label = xyz_path.stem.replace("h2_shell_snapshot_", "")
        ax_main = fig.add_subplot(row_gs[0, 0])
        ax_xy = fig.add_subplot(row_gs[0, 1])
        ax_xz = fig.add_subplot(row_gs[0, 2])
        ax_yz = fig.add_subplot(row_gs[0, 3])
        ax_legend = fig.add_subplot(row_gs[1, 2:4])
        _plot_main_view(ax_main, atoms, comment, "ABC"[row_index], step_label)
        _plot_projection(ax_xy, atoms, "xy")
        _plot_projection(ax_xz, atoms, "xz")
        _plot_projection(ax_yz, atoms, "yz")
        _plot_projection_legend(ax_legend)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_snapshots(
    *,
    xyz_paths: list[Path],
    output: Path,
    renderer: str = "auto",
) -> Path:
    """Generate the solvent-shell snapshot figure."""
    if not xyz_paths:
        raise ValueError("at least one XYZ snapshot is required")
    xyz_paths = xyz_paths[:3]
    if renderer == "ase":
        return _render_with_ase(xyz_paths, output)
    if renderer in {"auto", "pymol"}:
        if _render_with_pymol(xyz_paths, output):
            return output
        if renderer == "pymol":
            raise RuntimeError("PyMOL rendering requested but failed")
    return _render_with_ase(xyz_paths, output)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--xyz-glob", default="h2_shell_snapshot_step*.xyz")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--renderer", choices=["ase", "auto", "pymol"], default="ase")
    args = parser.parse_args(argv)

    xyz_paths = sorted(Path(path) for path in glob.glob(str(args.input_dir / args.xyz_glob)))
    output = plot_snapshots(xyz_paths=xyz_paths, output=args.output, renderer=args.renderer)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
