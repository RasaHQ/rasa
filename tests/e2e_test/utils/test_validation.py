import platform
import typing
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest
from structlog.testing import capture_logs

from rasa.e2e_test.e2e_test_case import TestCase
from rasa.e2e_test.utils.validation import (
    validate_model_path,
    validate_path_to_test_cases,
    validate_test_case,
    validate_test_case_fixtures,
    validate_test_case_metadata,
)
from rasa.shared.constants import DEFAULT_MODELS_PATH

if typing.TYPE_CHECKING:
    from rasa.e2e_test.e2e_test_case import Fixture, Metadata


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
            validate_test_case(test_case, [], {}, {})


@pytest.mark.parametrize(
    "fixture_names, fixtures, expected_warnings, expected_all_good",
    [
        # No fixtures referenced or defined
        (
            None,
            {},
            [],
            True,
        ),
        # No fixtures referenced at all
        (
            None,
            {"fixture_1": MagicMock(), "fixture_2": MagicMock()},
            [],
            True,
        ),
        # All fixtures referenced are defined
        (
            ["fixture_1", "fixture_2"],
            {"fixture_1": MagicMock(), "fixture_2": MagicMock()},
            [],
            True,
        ),
        # Some fixtures referenced are not defined
        (
            ["fixture_1", "fixture_3"],
            {"fixture_1": MagicMock(), "fixture_2": MagicMock()},
            [
                {
                    "event_info": (
                        "Fixture 'fixture_3' referenced in the "
                        "test case 'test_case_name' is not defined."
                    ),
                    "event": "validation.validate_test_case_fixtures",
                    "log_level": "error",
                }
            ],
            False,
        ),
        # Fixtures referenced with no defined fixtures
        (
            ["fixture_1", "fixture_2"],
            {},
            [
                {
                    "event_info": (
                        "Fixture 'fixture_1' referenced in the "
                        "test case 'test_case_name' is not defined."
                    ),
                    "event": "validation.validate_test_case_fixtures",
                    "log_level": "error",
                },
                {
                    "event_info": (
                        "Fixture 'fixture_2' referenced in the "
                        "test case 'test_case_name' is not defined."
                    ),
                    "event": "validation.validate_test_case_fixtures",
                    "log_level": "error",
                },
            ],
            False,
        ),
    ],
)
def test_validate_test_case_fixtures(
    fixture_names: Optional[List[str]],
    fixtures: Dict[str, "Fixture"],
    expected_warnings: List[Dict[str, Any]],
    expected_all_good: bool,
):
    test_case = TestCase(name="test_case_name", steps=[], fixture_names=fixture_names)
    with capture_logs() as caplog:
        all_good = validate_test_case_fixtures(test_case, fixtures)
        assert all_good == expected_all_good
        assert caplog == expected_warnings


@pytest.mark.parametrize(
    "metadata_name, metadata, expected_warnings, expected_all_good",
    [
        # No metadata referenced or defined
        (
            None,
            {},
            [],
            True,
        ),
        # No metadata referenced at all
        (
            None,
            {"metadata_1": MagicMock()},
            [],
            True,
        ),
        # Metadata referenced is defined
        (
            "metadata_1",
            {"metadata_1": MagicMock()},
            [],
            True,
        ),
        # Metadata referenced is not defined
        (
            "metadata_1",
            {},
            [
                {
                    "event_info": (
                        "Metadata 'metadata_1' referenced in the "
                        "test case 'test_case_name' is not defined."
                    ),
                    "event": (
                        "validation.validate_test_case_metadata.test_case_metadata"
                    ),
                    "log_level": "error",
                }
            ],
            False,
        ),
    ],
)
def test_validate_test_case_metadata(
    metadata_name: Optional[str],
    metadata: Dict[str, "Metadata"],
    expected_warnings: List[Dict[str, Any]],
    expected_all_good: bool,
):
    test_case = TestCase(name="test_case_name", steps=[], metadata_name=metadata_name)
    with capture_logs() as caplog:
        all_good = validate_test_case_metadata(test_case, metadata)
        assert all_good == expected_all_good
        assert caplog == expected_warnings


@pytest.mark.parametrize(
    "metadata_name, metadata, expected_warnings, expected_all_good",
    [
        # No metadata referenced or defined
        (
            None,
            {},
            [],
            True,
        ),
        # No metadata referenced at all
        (
            None,
            {"metadata_1": MagicMock()},
            [],
            True,
        ),
        # Metadata referenced is defined
        (
            "metadata_1",
            {"metadata_1": MagicMock()},
            [],
            True,
        ),
        # Metadata referenced is not defined
        (
            "metadata_1",
            {},
            [
                {
                    "event_info": (
                        "Metadata 'metadata_1' referenced in the step of the "
                        "test case 'test_case_name' is not defined."
                    ),
                    "event": "validation.validate_test_case_metadata.step_metadata",
                    "log_level": "error",
                }
            ],
            False,
        ),
    ],
)
def test_validate_test_case_metadata_step(
    metadata_name: Optional[str],
    metadata: Dict[str, "Metadata"],
    expected_warnings: List[Dict[str, Any]],
    expected_all_good: bool,
):
    step = MagicMock(metadata_name=metadata_name)
    test_case = TestCase(name="test_case_name", steps=[step], metadata_name=None)
    with capture_logs() as caplog:
        all_good = validate_test_case_metadata(test_case, metadata)
        assert all_good == expected_all_good
        assert caplog == expected_warnings
