import shutil
import subprocess
from pathlib import Path

import pytest


def test_api_key_copy_browser_behaviour():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required to execute browser helper tests")
    repository = Path(__file__).resolve().parents[2]
    subprocess.run(
        [node, "tests/js/test_api_key_copy.js"],
        cwd=repository,
        check=True,
        text=True,
        capture_output=True,
    )
