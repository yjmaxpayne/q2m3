"""Update and validate Chinese Sphinx catalogs against the current gettext output."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from babel.messages.catalog import Catalog
from babel.messages.pofile import read_po, write_po

ROOT = Path(__file__).resolve().parent
AUTHOR = "Ye Jun <yjmaxpayne@hotmail.com>"
CATALOG_HEADER = (
    "# Simplified Chinese translations for q2m3.\n"
    f"# Copyright (c) 2026 {AUTHOR}\n"
    "# Distributed under the MIT license.\n"
    f"# {AUTHOR}, 2026."
)


def read_catalog(path: Path, locale: str | None = None) -> Catalog:
    with path.open("rb") as stream:
        return read_po(stream, locale=locale, abort_invalid=True)


def template_paths(pot_dir: Path) -> dict[Path, Path]:
    return {p.relative_to(pot_dir).with_suffix(".po"): p for p in pot_dir.rglob("*.pot")}


def source_catalogs(pot_dir: Path, ui_messages: tuple[str, ...]) -> dict[Path, Catalog]:
    catalogs = {name: read_catalog(path) for name, path in template_paths(pot_dir).items()}
    if catalogs and ui_messages:
        interface = catalogs.setdefault(Path("sphinx.po"), Catalog())
        for message in ui_messages:
            interface.add(message)
    return catalogs


def check_catalogs(pot_dir: Path, locale_dir: Path, ui_messages: tuple[str, ...] = ()) -> list[str]:
    """Return all missing, stale, fuzzy, invalid, or untranslated message errors."""
    templates = source_catalogs(pot_dir, ui_messages)
    if not templates:
        return [f"No POT catalogs found in {pot_dir}. Run make docs-gettext first."]
    errors = []
    for relative, source in sorted(templates.items()):
        target_path = locale_dir / relative
        if not target_path.is_file():
            errors.append(f"Missing catalog: {target_path}")
            continue
        target = read_catalog(target_path, "zh_CN")
        if target.fuzzy:
            errors.append(f"{relative}: fuzzy catalog header")
        expected = {m.id for m in source if m.id}
        actual = {m.id for m in target if m.id}
        for message in sorted(expected - actual, key=str):
            errors.append(f"{relative}: missing message {message!r}")
        for message in sorted(actual - expected, key=str):
            errors.append(f"{relative}: stale message {message!r}")
        for message in target:
            if not message.id:
                continue
            if message.fuzzy:
                errors.append(f"{relative}: fuzzy message {message.id!r}")
            if isinstance(message.id, tuple):
                values = message.string if isinstance(message.string, tuple | list) else ()
                complete = len(values) == target.num_plurals and all(v.strip() for v in values)
            else:
                complete = isinstance(message.string, str) and bool(message.string.strip())
            if not complete:
                errors.append(f"{relative}: empty or incomplete translation {message.id!r}")
        for message, problems in target.check():
            for problem in problems:
                errors.append(f"{relative}: {message.id!r}: {problem}")
    for path in sorted(locale_dir.rglob("*.po")):
        if path.relative_to(locale_dir) not in templates:
            errors.append(f"Stale catalog: {path}")
    return errors


def update_catalogs(pot_dir: Path, locale_dir: Path, ui_messages: tuple[str, ...] = ()) -> None:
    """Preserve exact reviewed messages. Leave new or changed messages untranslated."""
    templates = source_catalogs(pot_dir, ui_messages)
    if not templates:
        raise ValueError(f"No POT catalogs found in {pot_dir}. Run make docs-gettext first.")
    for relative, source in sorted(templates.items()):
        path = locale_dir / relative
        catalog = read_catalog(path, "zh_CN") if path.exists() else Catalog(locale="zh_CN")
        catalog.update(source, no_fuzzy_matching=True)
        catalog.project = "q2m3"
        catalog.version = source.version
        catalog.copyright_holder = AUTHOR
        catalog.msgid_bugs_address = "yjmaxpayne@hotmail.com"
        catalog.last_translator = AUTHOR
        catalog.revision_date = datetime.now(UTC)
        catalog.language_team = "Simplified Chinese"
        catalog.header_comment = CATALOG_HEADER
        catalog.fuzzy = False
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            write_po(stream, catalog, width=100, omit_header=False, include_previous=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("update", "check"))
    parser.add_argument("--pot-dir", type=Path, default=ROOT / "build/gettext")
    parser.add_argument("--locale-dir", type=Path, default=ROOT / "locales/zh_CN/LC_MESSAGES")
    args = parser.parse_args()
    ui_messages = tuple(json.loads((ROOT / "ui-messages.json").read_text()))
    if args.command == "update":
        update_catalogs(args.pot_dir, args.locale_dir, ui_messages)
        print("Updated Chinese catalogs. Review new messages before building the site.")
        return 0
    errors = check_catalogs(args.pot_dir, args.locale_dir, ui_messages)
    for error in errors:
        print(error)
    if errors:
        print(f"Translation check failed: {len(errors)} error(s).")
        return 1
    print(f"Translation check passed: {len(template_paths(args.pot_dir))} catalogs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
