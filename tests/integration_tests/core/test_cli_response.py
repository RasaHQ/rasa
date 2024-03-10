from pathlib import Path
import pytest
from typing import Callable
import shutil
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


def test_rasa_validate_debug_no_errors(
    run: Callable[..., RunResult], request: FixtureRequest
):
    # Test captures the subprocess output for the command run
    # validates that the data in 'data/test/test_integration' throws no cli errors
    # in 'debug' mode
    test_data_dir = Path(request.config.rootdir, "data", "test", "test_integration")
    test_config_dir = Path(request.config.rootdir, "data", "test_config")
    source_file = (test_data_dir).absolute()
    domain_file = (test_data_dir / "domain.yml").absolute()
    config_file = (test_config_dir / "config_unique_assistant_id.yml").absolute()
    result = run(
        "data",
        "validate",
        "--data",
        str(source_file),
        "-d",
        str(domain_file),
        "-c",
        str(config_file),
        "--debug",
    )
    assert result.ret == 0
    assert "DEBUG" in str(result.stderr)
    output_text = "".join(result.outlines)
    assert "Rasa Open Source reports anonymous usage telemetry"
    "to help improve the product" in output_text
    assert "for all its users." in output_text
    assert "If you'd like to opt-out,"
    "you can use `rasa telemetry disable`." in output_text
    assert "To learn more, check out"
    "https://rasa.com/docs/rasa/telemetry/telemetry." in output_text


def test_rasa_validate_debug_with_errors(
    run: Callable[..., RunResult], request: FixtureRequest
):
    # Test captures the subprocess output for the command run
    # validates that the data in 'data/test/test_integration_incorrect'
    # throws cli errors about intent "greet"
    # in 'debug' mode
    err = (
        "UserWarning: The intent 'greet' is listed in the domain file, "
        "but is not found in the NLU training data."
    )
    test_data_dir = Path(request.config.rootdir, "data", "test", "test_integration_err")
    test_config_dir = Path(request.config.rootdir, "data", "test_config")
    source_file = (test_data_dir).absolute()
    domain_file = (test_data_dir / "domain.yml").absolute()
    config_file = (test_config_dir / "config_unique_assistant_id.yml").absolute()
    result = run(
        "data",
        "validate",
        "--data",
        str(source_file),
        "-d",
        str(domain_file),
        "-c",
        str(config_file),
        "--fail-on-warnings",
        "--debug",
    )
    assert result.ret == 1
    assert err in str(result.stderr)


def test_rasa_validate_verbose_no_errors(
    run: Callable[..., RunResult], request: FixtureRequest
):
    # Test captures the subprocess output for the command run
    # and validates that the data in 'data/test/test_integration' throws no cli errors
    # in 'verbose' mode
    test_data_dir = Path(request.config.rootdir, "data", "test", "test_integration")
    test_config_dir = Path(request.config.rootdir, "data", "test_config")
    source_file = (test_data_dir).absolute()
    domain_file = (test_data_dir / "domain.yml").absolute()
    config_file = (test_config_dir / "config_unique_assistant_id.yml").absolute()
    result = run(
        "data",
        "validate",
        "--data",
        str(source_file),
        "-d",
        str(domain_file),
        "-c",
        str(config_file),
        "--verbose",
    )
    assert result.ret == 0
    assert "INFO" in str(result.stderr)
    output_text = "".join(result.outlines)
    assert "Rasa Open Source reports anonymous usage telemetry"
    "to help improve the product" in output_text
    assert "for all its users." in output_text
    assert "If you'd like to opt-out,"
    "you can use `rasa telemetry disable`." in output_text
    assert "To learn more, check out"
    "https://rasa.com/docs/rasa/telemetry/telemetry." in output_text


def test_rasa_validate_quiet_no_errors(
    run: Callable[..., RunResult], request: FixtureRequest
):
    # Test captures the subprocess output for the command run
    # and validates that the data in 'data/test/test_integration' throws no cli errors
    # in 'quiet' mode
    test_data_dir = Path(request.config.rootdir, "data", "test", "test_integration")
    test_config_dir = Path(request.config.rootdir, "data", "test_config")
    source_file = (test_data_dir).absolute()
    domain_file = (test_data_dir / "domain.yml").absolute()
    config_file = (test_config_dir / "config_unique_assistant_id.yml").absolute()
    result = run(
        "data",
        "validate",
        "--data",
        str(source_file),
        "-d",
        str(domain_file),
        "-c",
        str(config_file),
        "--quiet",
    )
    assert result.ret == 0
    output_text = "".join(result.outlines)
    assert "Rasa Open Source reports anonymous usage telemetry"
    "to help improve the product" in output_text
    assert "for all its users." in output_text
    assert "If you'd like to opt-out,"
    "you can use `rasa telemetry disable`." in output_text
    assert "To learn more, check out"
    "https://rasa.com/docs/rasa/telemetry/telemetry." in output_text


def test_rasa_validate_null_active_loop_no_errors(
    run: Callable[..., RunResult], request: FixtureRequest
):
    # Test captures the subprocess output for the command run
    # and validates that the data in 'data/test/test_integration' throws no cli errors

    test_data_dir = Path(request.config.rootdir, "data", "test", "test_integration")
    test_config_dir = Path(request.config.rootdir, "data", "test_config")
    source_file = (test_data_dir / "data").absolute()
    domain_file = (test_data_dir / "domain.yml").absolute()
    config_file = (test_config_dir / "config_unique_assistant_id.yml").absolute()
    result = run(
        "data",
        "validate",
        "--data",
        str(source_file),
        "-d",
        str(domain_file),
        "-c",
        str(config_file),
    )
    assert result.ret == 0

    stderr_text = str(result.stderr)
    assert "INFO" in stderr_text
    assert "Validating intents..." in stderr_text
    assert "Validating utterances..." in stderr_text
    assert "Story structure validation..." in stderr_text
    assert "Validating utterances..." in stderr_text
    assert "Considering all preceding turns for conflict analysis." in stderr_text
    assert "No story structure conflicts found." in stderr_text
