import re
import subprocess
import sys

import pytest

import elicitor
from elicitor.cli import main


def test_version_is_semver():
    assert re.fullmatch(r"\d+\.\d+\.\d+", elicitor.__version__)


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == f"elicitor {elicitor.__version__}"


def test_no_arguments_prints_help(capsys):
    assert main([]) == 0
    assert capsys.readouterr().out.startswith("usage: elicitor")


def test_module_entry_point():
    result = subprocess.run(
        [sys.executable, "-m", "elicitor", "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == f"elicitor {elicitor.__version__}"
