from __future__ import annotations

import importlib.util
import os
import posixpath
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as metadata_version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

project = "q2m3"
author = "Ye Jun <yjmaxpayne@hotmail.com>"
copyright = "2026, Ye Jun <yjmaxpayne@hotmail.com>"
first_release = "0.1.0"


def _resolve_version() -> str:
    pinned = os.environ.get("Q2M3_DOCS_VERSION", "").strip()
    if pinned:
        return pinned

    try:
        return metadata_version(project)
    except PackageNotFoundError:
        pass

    version_module = ROOT / "src" / "q2m3" / "version.py"
    try:
        spec = importlib.util.spec_from_file_location("_q2m3_version", version_module)
        if spec is None or spec.loader is None:
            return "0.0.0"
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.get_version()
    except Exception:
        return "0.0.0"


version = _resolve_version()
release = version
html_context = {
    "display_version": first_release if release.startswith(f"{first_release}.dev") else release,
    "documentation_redirects": {
        "core-concepts": "user-guide/core-concepts",
        "sqd": "user-guide/sqd",
    },
}

# Keep published entry URLs after moving their sources into the user guide.
html_additional_pages = {name: "redirect.html" for name in html_context["documentation_redirects"]}

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinxcontrib.mermaid",
]

source_suffix = {".rst": "restructuredtext"}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

language = "en"
locale_dirs = ["../locales"]
gettext_compact = False
gettext_uuid = False
# These URL prefixes are independent of the HTML language tag (zh-CN).
language_paths = {"en": "", "zh_CN": "zh_CN/"}

nitpicky = False

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_typehints_format = "short"

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False

doctest_global_setup = """
import numpy as np
from q2m3.core.qpe import QPEEngine
from q2m3.constants import HARTREE_TO_KCAL_MOL, KCAL_TO_HARTREE
from q2m3.interfaces import PySCFPennyLaneConverter
"""

copybutton_prompt_text = r"^\$ "
copybutton_prompt_is_regexp = True

html_theme = "furo"
html_title = "q2m3 Documentation"
html_logo = "_static/logo.svg"
html_static_path = ["_static"]
html_css_files = ["css/style.css"]
# Ensure GitHub Pages serves _static/_*.js etc. untouched (no Jekyll filtering).
html_extra_path = [".nojekyll"]

html_theme_options = {
    "sidebar_hide_name": False,
    "light_logo": "",
    "dark_logo": "",
}


def _language_links(app, pagename, templatename, context, doctree):
    uri = app.builder.get_target_uri(pagename)
    current = app.config.language_paths[app.config.language] + uri
    context["language_links"] = {
        code: posixpath.relpath(prefix + uri, posixpath.dirname(current) or ".")
        for code, prefix in app.config.language_paths.items()
    }


def _language_title(app, config):
    if config.language == "zh_CN":
        config.html_title = "q2m3 文档"


def _viewcode_alias_module(app, modname, fullname):
    # Keep these re-exports from replacing the canonical core API backlinks.
    if modname == "q2m3.solvation.circuit_builder" and fullname in {
        "decompose_hamiltonian",
        "build_operator_index_map",
    }:
        return modname
    return None


def _chinese_search_language():
    from sphinx.search.zh import SearchChinese

    class ChineseSearch(SearchChinese):
        # Use matching identity stemmers for Chinese words and exact API names.
        js_stemmer_rawcode = ""

        def stem(self, word):
            return word.lower()

    return ChineseSearch


def setup(app):
    app.add_config_value("language_paths", language_paths, "html")
    app.connect("config-inited", _language_title)
    app.connect("html-page-context", _language_links)
    app.connect("viewcode-follow-imported", _viewcode_alias_module)
    app.add_search_language(_chinese_search_language())
