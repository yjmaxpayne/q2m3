"""The fragment tutorials: the calculator page runs the water-dimer workflow in at most
five lines of code, and the report-reading page explains the energies from that run."""

from __future__ import annotations

import ast
import doctest
import re
import textwrap
from pathlib import Path

TUTORIALS = Path(__file__).resolve().parents[2] / "doc/source/tutorials"
TUTORIAL = TUTORIALS / "fragment-interaction.rst"
READING_TUTORIAL = TUTORIALS / "fragment-report.rst"
MAX_CODE_LINES = 5
_BLOCK = re.compile(
    r"^\.\. (doctest|testsetup|testcleanup)::[^\n]*\n\n((?:[ \t]+[^\n]*\n|\n)+)",
    re.M,
)
_FILE_REFERENCE = re.compile(
    r":download:`(?:[^`<]*<)?([^`<>]+)>?`|^\.\. literalinclude::[ \t]+(\S+)", re.M
)


def _blocks(text: str, kind: str) -> list[str]:
    return [textwrap.dedent(body) for name, body in _BLOCK.findall(text) if name == kind]


def code_lines(block: str) -> list[str]:
    """Return the physical source lines of the non-import statements of a doctest block.

    Blank lines and comment-only lines do not count.
    """
    lines = []
    for example in doctest.DocTestParser().get_examples(block):
        source = example.source.splitlines()
        for node in ast.parse(example.source).body:
            if isinstance(node, ast.Import | ast.ImportFrom):
                continue
            for line in source[node.lineno - 1 : node.end_lineno]:
                if line.strip() and not line.strip().startswith("#"):
                    lines.append(line)
    return lines


def test_imports_blank_and_comment_lines_do_not_count():
    block = ">>> import os\n>>> from q2m3 import run\n>>> # a comment\n>>> x = 1\n"
    assert code_lines(block) == ["x = 1"]


def test_each_physical_line_of_a_statement_counts():
    block = ">>> x = [\n...     1,\n...     # a comment\n...\n...     2,\n... ]\n"
    assert len(code_lines(block)) == 4


def test_tutorial_workflow_fits_in_five_code_lines():
    text = TUTORIAL.read_text()
    code = [line for block in _blocks(text, "doctest") for line in code_lines(block)]
    assert any("Molecule.from_xyz(" in line for line in code)
    assert any("run(" in line for line in code)
    assert len(code) <= MAX_CODE_LINES, code


def test_hidden_setup_runs_no_workflow_code():
    text = TUTORIAL.read_text()
    hidden = _blocks(text, "testsetup") + _blocks(text, "testcleanup")
    assert not any("q2m3" in block for block in hidden)


def file_references(text: str) -> list[str]:
    """Return the local files that a page links with ``:download:`` or shows with
    ``literalinclude``, relative to the page."""
    return [download or included for download, included in _FILE_REFERENCE.findall(text)]


def _report_want() -> str:
    """Return the expected report output of the calculator tutorial's doctest."""
    examples = [
        example
        for block in _blocks(TUTORIAL.read_text(), "doctest")
        for example in doctest.DocTestParser().get_examples(block)
    ]
    (report,) = [example for example in examples if "render_report(" in example.source]
    return report.want


def test_download_links_and_included_files_are_references():
    text = (
        "See :download:`report.md`.\n"
        "Get :download:`the report <dir/r.md>`.\n"
        ".. literalinclude:: shown.md\n   :language: markdown\n"
    )
    assert file_references(text) == ["report.md", "dir/r.md", "shown.md"]


def test_calculator_tutorial_references_existing_files():
    references = file_references(TUTORIAL.read_text())
    assert references
    missing = [name for name in references if not (TUTORIALS / name).is_file()]
    assert not missing, missing


def test_reading_tutorial_matches_the_calculator_run():
    # Sphinx checks these expected energies against a real calculation.
    expected = dict(re.findall(r"^\| ([^|]+) \| [^|]+ \| ([^|]+) \| \|$", _report_want(), re.M))
    reading = READING_TUTORIAL.read_text()
    actual = dict(
        re.findall(
            r"^   \* - (HF|MP2|CASCI|SQD|CCSD\(T\))\n     - [^\n]+\n     - ([^\n]+)",
            reading.replace("−", "-"),
            re.M,
        )
    )
    assert len(expected) == 5
    assert actual == expected
    assert ":doc:`Fragment Interaction Energy <fragment-interaction>`" in reading
