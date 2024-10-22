import platform
from pathlib import Path

import pytest
from structlog.testing import capture_logs

from rasa.e2e_test.utils.validation import (
    validate_model_path,
    validate_path_to_test_cases,
    validate_test_case,
)
from rasa.shared.constants import DEFAULT_MODELS_PATH


def test_validate_model_path(tmp_path: Path) -> None:
    model_path = tmp_path / "model.tar.gz"
    model_path.touch()
    assert validate_model_path(str(model_path), "model", DEFAULT_MODELS_PATH) == str(
        model_path
    )


def test_validate_model_path_path_not_exists(tmp_path: Path) -> None:
    model_path = tmp_path / "model.tar.gz"
    default = tmp_path / DEFAULT_MODELS_PATH
    match_msg = (
        f"The provided model path '{model_path!s}' could not be found. "
        f"Using default location '{default!s}' instead."
    )
    if platform.system() == "Windows":
        # Windows uses backslashes in paths
        match_msg = match_msg.replace("\\", "\\\\")

    with pytest.warns(UserWarning, match=match_msg):
        assert validate_model_path(str(model_path), "model", default) == default


def test_validate_model_path_with_none(tmp_path: Path) -> None:
    parameter = "model"
    default = tmp_path / DEFAULT_MODELS_PATH
    with capture_logs() as logs:
        assert validate_model_path(None, parameter, default) == default

    log_msg = (
        f"Parameter '{parameter}' is not set. "
        f"Using default location '{default}' instead."
    )
    assert log_msg in logs[0]["message"]


def test_validate_path_to_test_cases(tmp_path: Path) -> None:
    """Test that a path to test cases which doesn't exist is validated correctly.

    The tested function should raise a UserWarning and exit the program.
    """
    path_to_test_cases = tmp_path / "test_cases.yml"

    match_msg = f"Path to test cases does not exist: {path_to_test_cases!s}."

    if platform.system() == "Windows":
        # Windows uses backslashes in paths
        match_msg = match_msg.replace("\\", "\\\\")

    with pytest.warns(UserWarning, match=match_msg):
        with pytest.raises(SystemExit):
            validate_path_to_test_cases(str(path_to_test_cases))


def test_validate_test_case() -> None:
    """Test that a path to a test case which doesn't exist is validated correctly.

    The tested function should raise a UserWarning and exit the program.
    """
    test_case = "test_case1"
    match_msg = f"Test case does not exist: {test_case!s}."

    with pytest.warns(UserWarning, match=match_msg):
        with pytest.raises(SystemExit):
            validate_test_case(test_case, [])
