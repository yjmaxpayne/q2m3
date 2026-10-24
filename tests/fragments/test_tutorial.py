"""The calculator tutorial runs the water-dimer workflow in at most five lines of code."""

from __future__ import annotations

import ast
import doctest
import re
from pathlib import Path

TUTORIAL = Path(__file__).resolve().parents[2] / "doc/source/tutorials/fragment-interaction.md"
MAX_CODE_LINES = 5
_BLOCK = re.compile(r"^```\{(doctest|testsetup|testcleanup)\}[^\n]*\n(.*?)^```", re.M | re.S)


def _blocks(text: str, kind: str) -> list[str]:
    return [body for name, body in _BLOCK.findall(text) if name == kind]


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
