from argparse import Namespace
from typing import List, Union
from unittest.mock import patch

import pytest

from rasa.studio.utils import validate_argument_paths


@pytest.mark.parametrize(
    "arg_name, path_value",
    [
        ("domain", "invalid_domain.yml"),
        ("config", "invalid_config.yml"),
        ("endpoints", "invalid_endpoints.yml"),
        ("data", "invalid_data_folder"),
        ("data", ["valid_data_folder"]),
        ("data", ["valid_data_file.yml", "another_valid_data_file.yml"]),
    ],
)
def test_validate_argument_paths_invalid_paths_raises(
    arg_name: str, path_value: Union[str, List[str]]
):
    args = Namespace(**{arg_name: path_value})
    with (
        patch("pathlib.Path.exists", return_value=False),
        patch("rasa.shared.utils.cli.print_error_and_exit") as mock_exit,
    ):
        validate_argument_paths(args)
        mock_exit.assert_called_once()


@pytest.mark.parametrize(
    "arg_name, path_values",
    [
        ("data", ["valid_data_folder"]),
        ("data", ["valid_data_file.yml", "another_valid_data_file.yml"]),
    ],
)
def test_validate_argument_paths_accepts_existing_paths(
    arg_name: str, path_values: List[str]
) -> None:
    args = Namespace(**{arg_name: path_values})
    with (
        patch("pathlib.Path.exists", return_value=True),
        patch("rasa.shared.utils.cli.print_error_and_exit") as mock_exit,
    ):
        validate_argument_paths(args)
        mock_exit.assert_not_called()


def test_validate_argument_paths_aggregates_errors() -> None:
    args = Namespace(
        domain="missing_domain.yml",
        data=["missing_data_1", "missing_data_2"],
    )

    with (
        patch("pathlib.Path.exists", return_value=False),
        patch("rasa.shared.utils.cli.print_error_and_exit") as mock_exit,
    ):
        validate_argument_paths(args)
        mock_exit.assert_called_once()

        error_msg = mock_exit.call_args[0][0]
        assert "missing_domain.yml" in error_msg
        assert "missing_data_1" in error_msg
        assert "missing_data_2" in error_msg
