from pathlib import Path
import sys
import pytest
from typing import Callable
import shutil
import subprocess
from _pytest.pytester import Pytester
from _pytest.pytester import RunResult
from _pytest.fixtures import FixtureRequest

# NOTE this will be extended to test cli logs at process run to validate log level
@pytest.fixture
def run(pytester: Pytester) -> Callable[..., RunResult]:
    def do_run(*args):
        args = [shutil.which("rasa")] + list(args)
        return pytester.run(*args)

    return do_run

def test_rasa_validate_debug_no_errors(run: Callable[..., RunResult], tmp_path: Path, request: FixtureRequest):
    # Test captures the subprocess output for the command run
    # validates that the data in 'data/test/test_integration' throws no cli errors
    # in 'debug' mode
    test_data_dir = Path(request.config.rootdir, "data", "test", "test_integration")
    source_file = (test_data_dir).absolute()
    domain_file = (test_data_dir / "domain.yml").absolute()
    result = run(
            "data",
            "validate",
            "--data",
            str(source_file),
            "-d",
            str(domain_file),
            "--debug",
        )
    assert result.ret == 0


def test_rasa_validate_verbose_no_errors(run: Callable[..., RunResult], tmp_path: Path, request: FixtureRequest):
    # Test captures the subprocess output for the command run
    # and validates that the data in 'data/test/test_integration' throws no cli errors
    # in 'verbose' mode
    test_data_dir = Path(request.config.rootdir, "data", "test", "test_integration")
    source_file = (test_data_dir).absolute()
    domain_file = (test_data_dir / "domain.yml").absolute()
    result = run(
             "data",
             "validate",
             "--data",
             str(source_file),
             "-d",
             str(domain_file),
             "--verbose",
            )
    assert result.ret == 0


def test_rasa_validate_quiet_no_errors(run: Callable[..., RunResult], tmp_path: Path, request: FixtureRequest):
    # Test captures the subprocess output for the command run
    # and validates that the data in 'data/test/test_integration' throws no cli errors
    # in 'quiet' mode
    test_data_dir = Path(request.config.rootdir, "data", "test", "test_integration")
    source_file = (test_data_dir).absolute()
    domain_file = (test_data_dir / "domain.yml").absolute()
    result = run(
             "data",
             "validate",
             "--data",
             str(source_file),
             "-d",
             str(domain_file),
             "--quiet",
             )
    assert result.ret == 0
