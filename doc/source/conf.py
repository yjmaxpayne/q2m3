from __future__ import annotations

import importlib.util
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
}

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinx_copybutton",
    "sphinxcontrib.mermaid",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

nitpicky = False

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_typehints_format = "short"

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False

myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
    "linkify",
]
myst_heading_anchors = 3
myst_linkify_fuzzy_links = False

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
html_static_path = ["_static"]
html_css_files = ["css/style.css"]

html_theme_options = {
    "sidebar_hide_name": False,
    "light_logo": "",
    "dark_logo": "",
}
