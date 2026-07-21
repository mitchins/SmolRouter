import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "script",
    ["tests/js/test_api_key_copy.js", "tests/js/test_project_ui_rendering.js"],
)
def test_browser_behaviour(script):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required to execute browser helper tests")
    repository = Path(__file__).resolve().parents[2]
    subprocess.run(
        [node, script],
        cwd=repository,
        check=True,
        text=True,
        capture_output=True,
    )
