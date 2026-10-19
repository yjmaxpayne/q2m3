"""Keep both published language trees usable and scientifically equivalent."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "doc_site", Path(__file__).resolve().parents[1] / "doc/check_site.py"
)
site = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(site)


@pytest.fixture
def bilingual_site(tmp_path):
    root = tmp_path / "html"
    for prefix, language, other in (
        ("", "en", "zh_CN/index.html"),
        ("zh_CN", "zh-CN", "../index.html"),
    ):
        page = root / prefix / "index.html"
        page.parent.mkdir(parents=True, exist_ok=True)
        page.write_text(
            f'<html lang="{language}"><a hreflang="en" href="'
            + ("index.html" if not prefix else other)
            + '">English</a><a hreflang="zh-CN" href="'
            + (other if not prefix else "index.html")
            + '">简体中文</a><article><pre>energy = -1.0</pre>'
            '<dt class="sig">compute(x)</dt><p id="result">Result</p>'
            '<a class="reference internal" href="#result">Result</a></article></html>'
        )
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"pages": ["index.html"]}))
    return root, baseline


def test_equivalent_language_trees_pass(bilingual_site):
    assert site.check_site(*bilingual_site) == []


@pytest.mark.parametrize(
    "damage", ["page", "code", "signature", "switch", "anchor", "asset", "reference", "formula"]
)
def test_broken_delivery_blocks_publication(bilingual_site, damage):
    root, baseline = bilingual_site
    path = root / "zh_CN/index.html"
    text = path.read_text()
    if damage == "page":
        path.unlink()
    else:
        if damage == "code":
            text = text.replace("energy = -1.0", "energy = 1.0")
        elif damage == "signature":
            text = text.replace("compute(x)", "compute(y)")
        elif damage == "switch":
            text = text.replace("../index.html", "index.html")
        elif damage == "anchor":
            text = text.replace('href="#result"', 'href="#missing"')
        elif damage == "asset":
            text += '<script src="_static/missing.js"></script>'
        elif damage == "reference":
            text = text.replace('href="#result"', 'href="#other"')
            text += '<p id="other">Other valid target</p>'
        else:
            text += '<div class="math">E = 1.0</div>'
        path.write_text(text)
    assert site.check_site(root, baseline), damage


@pytest.mark.parametrize(
    "broken_html",
    [
        '<img src="/q2m3/_static/missing.png">',
        '<a href="/q2m3/missing.html#broken">Missing</a>',
        '<button aria-label="Toggle Light / Dark / Auto color theme"></button>',
        "<footer>Made with Sphinx and Furo</footer>",
    ],
)
def test_missing_absolute_targets_and_english_interface_fail(bilingual_site, broken_html):
    root, baseline = bilingual_site
    path = root / "zh_CN/index.html"
    path.write_text(path.read_text() + broken_html)
    assert site.check_site(root, baseline)


def test_valid_site_absolute_target_passes(bilingual_site):
    root, baseline = bilingual_site
    path = root / "zh_CN/index.html"
    path.write_text(path.read_text() + '<a href="/q2m3/index.html#result">Result</a>')
    assert site.check_site(root, baseline) == []


def test_source_link_labels_can_translate_without_changing_code(bilingual_site):
    root, baseline = bilingual_site
    for prefix, label in (("", "[source]"), ("zh_CN", "[源代码]")):
        path = root / prefix / "index.html"
        text = path.read_text().replace(
            "compute(x)</dt>",
            'compute(x)<a class="reference internal" href="#result">'
            f'<span class="viewcode-link"><span>{label}</span></span></a></dt>',
        )
        path.write_text(text)
    assert site.check_site(root, baseline) == []


@pytest.mark.parametrize(
    "module,name,expected",
    [
        (
            "q2m3.solvation.circuit_builder",
            "decompose_hamiltonian",
            "q2m3.solvation.circuit_builder",
        ),
        (
            "q2m3.solvation.circuit_builder",
            "build_operator_index_map",
            "q2m3.solvation.circuit_builder",
        ),
        ("q2m3.core.hamiltonian_utils", "decompose_hamiltonian", None),
        ("q2m3.sqd", "LUCJConfig", None),
    ],
)
def test_viewcode_alias_fix_preserves_other_source_pages(module, name, expected):
    spec = importlib.util.spec_from_file_location(
        "doc_conf", Path(__file__).resolve().parents[1] / "doc/source/conf.py"
    )
    conf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conf)
    assert conf._viewcode_alias_module(None, module, name) == expected


def test_chinese_search_keeps_chinese_words_and_api_names():
    pytest.importorskip("sphinx")
    spec = importlib.util.spec_from_file_location(
        "doc_conf", Path(__file__).resolve().parents[1] / "doc/source/conf.py"
    )
    conf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conf)
    language = conf._chinese_search_language()({})
    assert language.stem("energies") == "energies"
    assert not language.js_stemmer_rawcode
    assert "stemWord" in language.js_stemmer_code
    pytest.importorskip("jieba")
    assert {"能量", "QPE"} <= set(language.split("能量 QPE"))
