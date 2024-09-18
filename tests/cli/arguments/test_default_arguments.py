import argparse
from typing import List

import pytest

from rasa.cli.arguments.default_arguments import add_skip_validation_flag


@pytest.mark.parametrize(
    "input_args, expected_skip_yaml_validation",
    [
        (
            [],
            [],
        ),
        (
            ["--skip-yaml-validation", "domain"],
            ["domain"],
        ),
    ],
)
def test_skip_yaml_validation_for_domain(
    input_args: List[str], expected_skip_yaml_validation: List[str]
) -> None:
    """Tests that --skip-yaml-validation is attached when add_skip_validation_flag is called."""  # noqa: E501
    parser = argparse.ArgumentParser()

    add_skip_validation_flag(parser)

    args = parser.parse_args(input_args)

    assert args.skip_yaml_validation == expected_skip_yaml_validation


def test_invalid_skip_yaml_validation() -> None:
    """Tests parsing of `--skip-yaml-validation` argument with invalid value."""
    parser = argparse.ArgumentParser()

    add_skip_validation_flag(parser)

    with pytest.raises(SystemExit):
        parser.parse_args(["--skip-yaml-validation", "invalid"])
