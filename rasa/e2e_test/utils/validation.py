import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Union

import structlog

import rasa.shared.utils.io
from rasa.e2e_test.constants import SCHEMA_FILE_PATH
from rasa.shared.utils.yaml import read_schema_file

if TYPE_CHECKING:
    from rasa.e2e_test.e2e_test_case import TestCase


structlogger = structlog.get_logger()


def validate_path_to_test_cases(path: str) -> None:
    """Validate that path to test cases exists."""
    if not Path(path).exists():
        rasa.shared.utils.io.raise_warning(
            f"Path to test cases does not exist: {path}. "
            f"Please provide a valid path to test cases. "
            f"Exiting...",
            UserWarning,
        )
        sys.exit(1)


def validate_test_case(test_case_name: str, input_test_cases: List["TestCase"]) -> None:
    """Validate that test case exists."""
    if test_case_name and not input_test_cases:
        rasa.shared.utils.io.raise_warning(
            f"Test case does not exist: {test_case_name}. "
            f"Please check for typos and provide a valid test case name. "
            f"Exiting...",
            UserWarning,
        )
        sys.exit(1)


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
        rasa.shared.utils.io.raise_warning(
            f"The provided model path '{model_path}' could not be found. "
            f"Using default location '{default}' instead.",
            UserWarning,
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
