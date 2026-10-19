"""Reject incomplete catalogs before publishing either documentation language."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from babel.messages.catalog import Catalog
from babel.messages.pofile import read_po, write_po

SPEC = importlib.util.spec_from_file_location(
    "doc_translations", Path(__file__).resolve().parents[1] / "doc/translations.py"
)
translations = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(translations)


def save(path, catalog):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        write_po(stream, catalog)


@pytest.fixture
def catalogs(tmp_path):
    pot = tmp_path / "gettext"
    locale = tmp_path / "locales/zh_CN/LC_MESSAGES"
    source = Catalog()
    source.add("Energy in Hartree.")
    source.add(("One sample", "%(num)d samples"))
    target = Catalog(locale="zh_CN")
    target.fuzzy = False
    target.add("Energy in Hartree.", "能量单位为 Hartree。")
    target.add(("One sample", "%(num)d samples"), ("%(num)d 个样本",))
    save(pot / "api/core.pot", source)
    save(locale / "api/core.po", target)
    return pot, locale


def test_complete_translation_passes(catalogs):
    assert translations.check_catalogs(*catalogs) == []


@pytest.mark.parametrize(
    "damage", ["catalog", "message", "stale", "empty", "plural", "fuzzy", "header"]
)
def test_incomplete_translation_blocks_publication(catalogs, damage):
    pot, locale = catalogs
    path = locale / "api/core.po"
    if damage == "catalog":
        path.unlink()
    else:
        with path.open("rb") as stream:
            catalog = read_po(stream, locale="zh_CN")
        if damage == "message":
            catalog.delete("Energy in Hartree.")
        elif damage == "stale":
            catalog.add("Removed statement", "已删除的说明")
        elif damage == "empty":
            catalog["Energy in Hartree."].string = ""
        elif damage == "plural":
            catalog["One sample"].string = ("",)
        elif damage == "fuzzy":
            catalog["Energy in Hartree."].flags.add("fuzzy")
        else:
            catalog.fuzzy = True
        save(path, catalog)
    assert translations.check_catalogs(pot, locale), damage


def test_empty_template_tree_fails_instead_of_skipping_checks(catalogs):
    pot, locale = catalogs
    (pot / "api/core.pot").unlink()
    assert translations.check_catalogs(pot, locale)


def test_update_preserves_reviewed_translation_and_adds_new_message(catalogs):
    pot, locale = catalogs
    with (pot / "api/core.pot").open("rb") as stream:
        catalog = read_po(stream)
    catalog.add("New condition.")
    save(pot / "api/core.pot", catalog)
    translations.update_catalogs(pot, locale)
    with (locale / "api/core.po").open("rb") as stream:
        updated = read_po(stream, locale="zh_CN")
    assert updated["Energy in Hartree."].string == "能量单位为 Hartree。"
    assert not updated["New condition."].string
    assert translations.check_catalogs(pot, locale)


def test_stale_catalog_is_rejected(catalogs):
    pot, locale = catalogs
    save(locale / "removed.po", Catalog(locale="zh_CN"))
    assert translations.check_catalogs(pot, locale)


def test_project_interface_inventory_is_required_even_without_template_messages(catalogs):
    pot, locale = catalogs
    assert translations.check_catalogs(pot, locale, ui_messages=("Search",))
    target = Catalog(locale="zh_CN")
    target.fuzzy = False
    target.add("Search", "搜索")
    save(locale / "sphinx.po", target)
    assert translations.check_catalogs(pot, locale, ui_messages=("Search",)) == []
