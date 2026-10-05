"""
The server and worker packages install as workspace members with their CLIs.
"""

import importlib
import subprocess
import sys

import pytest


@pytest.mark.parametrize("package", ["ml4paleo_server", "ml4paleo_worker"])
def test_cli_reports_version(package):
    version = importlib.import_module(package).__version__
    result = subprocess.run(
        [sys.executable, "-m", f"{package}.cli", "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert version in result.stdout
