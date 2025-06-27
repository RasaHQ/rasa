from argparse import Namespace
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
    ],
)
def test_validate_argument_paths_invalid_paths_raises(arg_name, path_value):
    args = Namespace(**{arg_name: path_value})
    with (
        patch("pathlib.Path.exists", return_value=False),
        patch("rasa.shared.utils.cli.print_error_and_exit") as mock_exit,
    ):
        validate_argument_paths(args)
        mock_exit.assert_called_once()
        assert path_value in mock_exit.call_args[0][0]
