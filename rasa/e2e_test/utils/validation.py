from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import structlog

from rasa.e2e_test.constants import SCHEMA_FILE_PATH
from rasa.e2e_test.e2e_test_case import Fixture, Metadata
from rasa.exceptions import ModelNotFound, ValidationError
from rasa.shared.utils.yaml import read_schema_file

if TYPE_CHECKING:
    from rasa.e2e_test.e2e_test_case import TestCase


structlogger = structlog.get_logger()


def validate_path_to_test_cases(path: str) -> None:
    """Validate that path to test cases exists."""
    if not Path(path).exists():
        raise ValidationError(
            code="e2e_test.utils.validation.invalid_test_case_path",
            event_info=f"Path to test cases does not exist: {path}. "
            f"Please provide a valid path to test cases. ",
        )


def validate_test_case(
    test_case_name: str,
    input_test_cases: List["TestCase"],
    fixtures: Dict[str, Fixture],
    metadata: Dict[str, Metadata],
) -> None:
    """
    Validate the test case, its fixtures, and metadata.

    Args:
        test_case_name (str): The name of the test case to validate.
        input_test_cases (List["TestCase"]): A list of test cases to validate.
        fixtures (Dict[str, Fixture]): A dictionary of defined fixtures.
        metadata (Dict[str, Metadata]): A dictionary of defined metadata.

    Raises:
        SystemExit: If the test case, fixtures, or metadata are not defined.
    """
    if test_case_name and not input_test_cases:
        raise ValidationError(
            code="e2e_test.utils.validation.invalid_test_case",
            event_info=f"Test case does not exist: {test_case_name}. "
            f"Please check for typos and provide a valid test case name. ",
        )

    all_good = True
    for test_case in input_test_cases:
        all_good_fixtures = validate_test_case_fixtures(test_case, fixtures)
        all_good_metadata = validate_test_case_metadata(test_case, metadata)
        all_good = all_good and all_good_fixtures and all_good_metadata

    if not all_good:
        raise ValidationError(
            code="e2e_test.utils.validation.missing_fixtures_or_metadata",
            event_info="Some fixtures and/or metadata are missing - "
            "see previous logs for additional details.",
        )


def validate_test_case_fixtures(
    test_case: "TestCase", fixtures: Dict[str, Fixture]
) -> bool:
    """Validates that the fixtures used in the test case are defined.

    Args:
        test_case (TestCase): The test case to validate.
        fixtures (Dict[str, Fixture]): A dictionary of defined fixtures.

    Returns:
        True if all fixtures used in the test case are defined, False otherwise.

    Raises:
        Logs an error if a fixture used in the test case is not defined.
    """
    all_good = True
    if not test_case.fixture_names:
        return all_good

    for fixture_name in test_case.fixture_names:
        if fixture_name not in fixtures:
            structlogger.error(
                "e2e_test.utils.validation.validate_test_case_fixtures",
                event_info=(
                    f"Fixture '{fixture_name}' referenced in the "
                    f"test case '{test_case.name}' is not defined."
                ),
            )
            all_good = False
    return all_good


def validate_test_case_metadata(
    test_case: "TestCase", metadata: Dict[str, Metadata]
) -> bool:
    """
    Validates that the metadata used in the test case and its steps are defined.

    Args:
        test_case (TestCase): The test case to validate.
        metadata (Dict[str, Metadata]): A dictionary of defined metadata.

    Returns:
        True if all fixtures used in the test case are defined, False otherwise.

    Raises:
        Logs an error if metadata used in the test case or its steps is not defined.
    """
    all_good = True
    if test_case.metadata_name and test_case.metadata_name not in metadata:
        structlogger.error(
            "e2e_test.utils.validation.validate_test_case_metadata.test_case_metadata",
            event_info=(
                f"Metadata '{test_case.metadata_name}' referenced in "
                f"the test case '{test_case.name}' is not defined."
            ),
        )
        all_good = False

    for step in test_case.steps:
        if step.metadata_name and step.metadata_name not in metadata:
            structlogger.error(
                "e2e_test.utils.validation.validate_test_case_metadata.step_metadata",
                event_info=(
                    f"Metadata '{step.metadata_name}' referenced in the "
                    f"step of the test case '{test_case.name}' is not defined."
                ),
            )
            all_good = False
    return all_good


def validate_model_path(model_path: Optional[str], parameter: str, default: str) -> str:
    """Validate the model path.

    Args:
        model_path: Path to the model.
        parameter: Name of the parameter.
        default: Default path to the model.

    Returns:
    Path to the model.
    """
    if model_path and Path(model_path).exists():
        return model_path

    if model_path and not Path(model_path).exists():
        raise ModelNotFound(
            f"The provided model path '{model_path}' could not be found. "
            "Provide an existing model path."
        )

    elif model_path is None:
        structlogger.info(
            "rasa.e2e_test.validate_model_path",
            message=f"Parameter '{parameter}' is not set. "
            f"Using default location '{default}' instead.",
        )

    Path(default).mkdir(exist_ok=True)
    return default


def read_e2e_test_schema() -> Union[List[Any], Dict[str, Any]]:
    """Read the schema for the e2e test files.

    Returns:
        The content of the schema.
    """
    return read_schema_file(SCHEMA_FILE_PATH)
