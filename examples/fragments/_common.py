"""Arguments, output files, and reports for the fragment examples."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

OUTPUT_ROOT = Path(__file__).resolve().parents[2] / "data/output/examples"
LAYERS = ("hf", "mp2", "casci", "sqd", "ccsd_t")
SUMMARY_FIELDS = (
    "fragment_id",
    "basis",
    "mode",
    "method",
    "delta_e_ha",
    "delta_e_kcal_mol",
    "null_reason",
    "warnings",
    "result_file",
)


def positive_float(value: str) -> float:
    """Parse a finite positive budget."""
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("Budget must be finite and positive")
    return number


def nonnegative_int(value: str) -> int:
    """Parse a nonnegative seed."""
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("Value must be nonnegative")
    return number


def positive_int(value: str) -> int:
    """Parse a positive shot count."""
    number = nonnegative_int(value)
    if number == 0:
        raise argparse.ArgumentTypeError("Value must be positive")
    return number


def parser(
    description: str, *, mode: str | None = None, choose_mode: bool = False, report: bool = False
) -> argparse.ArgumentParser:
    """Build CLI arguments without importing scientific backends."""
    cli = argparse.ArgumentParser(description=description)
    cli.add_argument("--output", type=Path, help="New exclusive run directory")
    if mode is not None:
        cli.set_defaults(mode=mode)
    if choose_mode:
        cli.add_argument("--mode", choices=("classical_only", "full"), default=mode)
    if mode == "full" or choose_mode:
        cli.add_argument("--seed", type=nonnegative_int, default=0)
        cli.add_argument("--shots", type=positive_int, default=100_000)
        cli.add_argument(
            "--wall-budget",
            type=positive_float,
            default=900.0,
            help="SQD budget per subsystem (seconds)",
        )
        cli.add_argument(
            "--rss-budget",
            type=positive_float,
            default=8192.0,
            help="SQD budget per subsystem (decimal MB)",
        )
        cli.add_argument(
            "--ccsd-t-wall-budget",
            type=positive_float,
            default=7200.0,
            help="Predicted summed CCSD(T) stage time limit (seconds)",
        )
        cli.add_argument(
            "--ccsd-t-rss-budget",
            type=positive_float,
            default=20000.0,
            help="Predicted CCSD(T) peak RSS limit (decimal MB)",
        )
    if report:
        cli.add_argument("--references", type=Path, help="Reference CSV covering both fragment IDs")
        cli.add_argument(
            "--ks-table", type=Path, help="JSON object mapping experimental labels to Ks"
        )
    return cli


def write_json(path: Path, value) -> None:
    """Save complete dataclasses with the public result serializer."""
    from q2m3.utils.io import save_json_results

    save_json_results(value, path)


def execute(args, name: str, calculate) -> int:
    """Create an exclusive directory and retain evidence after a failure."""
    directory = args.output or OUTPUT_ROOT / (
        f"{name}-{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    )
    directory = directory.resolve()
    try:
        directory.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        print(f"Cannot create run directory {directory}: {exc}", file=sys.stderr)
        return 1
    print(f"Output: {directory}", flush=True)
    started = time.perf_counter()
    record = {
        "example": name,
        "started_utc": datetime.now(UTC).isoformat(),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "status": "running",
    }
    write_json(directory / "run.json", record)
    status = 0
    try:
        calculate(args, directory)
        record["status"] = "success"
    except Exception as exc:
        status = 1
        record["status"] = "failed"
        write_json(
            directory / "failure.json",
            {
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        print(f"{name} failed: {exc}", file=sys.stderr)
    finally:
        record["elapsed_s"] = time.perf_counter() - started
        write_json(directory / "run.json", record)
    return status


def calculation_options(args, basis: str, labels: tuple[str, ...], threshold: float) -> dict:
    """Translate example arguments to existing workflow parameters."""
    options = dict(
        basis=basis,
        avas_labels=labels,
        avas_threshold=threshold,
        mode=args.mode,
        seed=getattr(args, "seed", 0),
    )
    if args.mode == "full":
        from examples.fragments._inputs import CRYSTAL_SETTINGS, WATER_SETTINGS
        from q2m3.fragments.references import CCSDTBudget
        from q2m3.sqd import LUCJConfig, ReferenceConfig

        # 2*n_occupied*n_virtual covers every real CCSD doubles factor.
        # Smaller monomer operators are padded with identity factors by ffsim.
        factor_slots = {WATER_SETTINGS: 24, CRYSTAL_SETTINGS: 36}[(basis, labels, threshold)]
        options.update(
            lucj=LUCJConfig(n_reps=factor_slots, shots=args.shots),
            reference=ReferenceConfig(
                wall_budget_s=args.wall_budget, rss_budget_mb=args.rss_budget
            ),
            ccsd_t_budget=CCSDTBudget(
                rss_mb=args.ccsd_t_rss_budget, wall_s=args.ccsd_t_wall_budget
            ),
        )
    return options


def save_input(directory: Path, molecule, options: dict) -> None:
    """Record the exact geometry, fragments, and requested calculation settings."""
    write_json(
        directory / "input.json",
        {
            "molecule": molecule,
            "geometry_sha256": molecule.geometry_sha256(),
            "options": options,
            "coordinate_unit": "Angstrom",
        },
    )
    for name, indices in zip(("A", "B"), molecule.fragments, strict=True):
        print(f"Fragment {name}, zero-based atom indices: {list(indices)}", flush=True)


def save_report(directory: Path, results, *, ks_table=None, result_files=None, failures=()) -> None:
    """Derive CSV and Markdown from the same complete results."""
    from q2m3.constants import HARTREE_TO_KCAL_MOL
    from q2m3.fragments import render_report

    result_files = result_files or ["result.json"] * len(results)
    with (directory / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for result, result_file in zip(results, result_files, strict=True):
            for layer in LAYERS:
                value = result.delta_e[layer]
                writer.writerow(
                    dict(
                        fragment_id=result.fragment_id,
                        basis=result.basis,
                        mode=result.mode,
                        method=layer,
                        delta_e_ha=value,
                        delta_e_kcal_mol=None if value is None else value * HARTREE_TO_KCAL_MOL,
                        null_reason=result.null_reasons.get(f"delta_e.{layer}", ""),
                        warnings=json.dumps(result_warnings(result), ensure_ascii=False),
                        result_file=result_file,
                    )
                )
    report = render_report(results, ks_table=ks_table)
    for result in results:
        warnings = result_warnings(result)
        if warnings:
            report += f"\n## 警告：{result.fragment_id}\n\n"
            report += "\n".join(f"- {warning}" for warning in warnings) + "\n"
        if result.mode == "full":
            from examples.fragments._accuracy import accuracy_markdown, assess_accuracy

            report += accuracy_markdown(result.fragment_id, assess_accuracy(result))
    if failures:
        report += "\n## 失败的计算\n\n"
        report += "\n".join(f"- {name}：见 `{name}/failure.json`。" for name in failures) + "\n"
    (directory / "report.md").write_text(report, encoding="utf-8")


def result_warnings(result) -> list[str]:
    """Include subsystem SQD warnings in the human-readable output."""
    warnings = list(result.warnings)
    for name, subsystem in result.subsystems.items():
        if subsystem.sqd is not None:
            warnings.extend(f"{name}: {warning}" for warning in subsystem.sqd.warnings)
    return warnings


def save_result(directory: Path, result):
    """Retain every result field and display each interaction energy."""
    write_json(directory / "result.json", result)
    result.validate()
    save_report(directory, [result])
    if result.mode == "full":
        from examples.fragments._accuracy import assess_accuracy

        evidence = assess_accuracy(result)
        write_json(directory / "accuracy.json", evidence)
        if not evidence["passed"]:
            raise ValueError("SQD accuracy acceptance failed; see accuracy.json and report.md")
    for layer, value in result.delta_e.items():
        detail = f"{value:.9f} Ha" if value is not None else result.null_reasons[f"delta_e.{layer}"]
        print(f"{result.fragment_id} / {layer}: {detail}", flush=True)
    return result


def load_ks(path: Path) -> dict[str, float]:
    """Read experimental labels and finite numeric values without inference."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or any(
        not key.strip()
        or any(c in key for c in "\n\r|")
        or isinstance(number, bool)
        or not isinstance(number, int | float)
        or not math.isfinite(number)
        for key, number in value.items()
    ):
        raise ValueError("Ks JSON must map nonempty single-line labels to finite numbers")
    return value
