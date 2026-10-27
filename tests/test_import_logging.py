"""Importing the SDK must not configure the host application's logging."""
import os
from pathlib import Path
import subprocess
import sys
import pytest


@pytest.mark.parametrize("api_key", [None, ""])
def test_import_does_not_configure_logging(tmp_path, api_key):
    sdk_root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(sdk_root)}
    if api_key is None:
        env.pop("OPENAI_API_KEY", None)
    else:
        env["OPENAI_API_KEY"] = api_key
    code = """
import logging
from pathlib import Path

root = logging.getLogger()
handler = logging.StreamHandler()
root.addHandler(handler)
root.setLevel(logging.WARNING)
def unexpected(*args, **kwargs):
    raise AssertionError('SDK import must not create log files or directories')
Path.mkdir = unexpected
logging.FileHandler = unexpected
import agentic_framework
assert root.handlers == [handler]
assert root.level == logging.WARNING
"""
    subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env, check=True)
