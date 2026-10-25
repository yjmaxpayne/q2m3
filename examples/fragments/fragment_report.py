"""Calculate both dimers sequentially and report energies with optional references and Ks."""

from __future__ import annotations

import argparse
import shutil

from examples.fragments import crystal_dimer_interaction, water_dimer_classical, water_dimer_full
from examples.fragments._common import execute, load_ks, parser, save_report, write_json
from examples.fragments._inputs import CRYSTAL_ID, CRYSTAL_SETTINGS, WATER_ID, WATER_SETTINGS


def calculate(args, directory) -> None:
    """Preserve each calculation and report every failed item."""
    from q2m3.fragments import import_references, render_report
    from q2m3.fragments.result import references_by_category

    water = water_dimer_full if args.mode == "full" else water_dimer_classical
    cases = (
        (WATER_ID, WATER_SETTINGS[0], water.calculate),
        (CRYSTAL_ID, CRYSTAL_SETTINGS[0], crystal_dimer_interaction.calculate),
    )
    references = {name: () for name, _, _ in cases}
    if args.references is not None:
        if args.references.suffix != ".csv":
            raise ValueError("--references requires a CSV covering both fragment IDs")
        source = directory / "references.csv"
        shutil.copyfile(args.references, source)
        for name, basis, _ in cases:
            references[name] = import_references(source, fragment_id=name, basis=basis)
            references_by_category(references[name])
    ks_table = None
    if args.ks_table is not None:
        source = directory / "ks-table.json"
        shutil.copyfile(args.ks_table, source)
        ks_table = load_ks(source)
        render_report([], ks_table=ks_table)
    results, files, failures = [], [], []
    for name, _, calculator in cases:
        child_args = argparse.Namespace(**(vars(args) | {"output": directory / name}))

        def collect(child_args, child_directory, *, calculator=calculator, name=name):
            result = calculator(child_args, child_directory, references=references[name])
            results.append(result)
            files.append(f"{name}/result.json")

        if execute(child_args, name, collect):
            failures.append(name)
    write_json(directory / "results.json", {"results": results, "failures": failures})
    save_report(directory, results, ks_table=ks_table, result_files=files, failures=failures)
    if failures:
        raise RuntimeError(f"Failed calculations: {', '.join(failures)}")


def main(argv=None) -> int:
    """Run both examples and return nonzero if any item fails."""
    args = parser(__doc__, mode="classical_only", choose_mode=True, report=True).parse_args(argv)
    return execute(args, "fragment-report", calculate)


if __name__ == "__main__":
    raise SystemExit(main())
