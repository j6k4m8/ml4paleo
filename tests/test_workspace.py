"""
The server and worker packages install as workspace members with their CLIs.
"""

import subprocess
import sys

import pytest


@pytest.mark.parametrize("module", ["ml4paleo_server.cli", "ml4paleo_worker.cli"])
def test_cli_reports_version(module):
    result = subprocess.run(
        [sys.executable, "-m", module, "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "0.2.0.dev0" in result.stdout
