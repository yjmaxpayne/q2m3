"""
Fixtures for profiling tests.
Restores working directory after each test to prevent os.chdir() pollution
from ir_output_dir() context manager.
"""

import os

import pytest


@pytest.fixture(autouse=True)
def restore_cwd():
    """Restore working directory after each test. Required for ir_output_dir tests."""
    original = os.getcwd()
    yield
    os.chdir(original)
