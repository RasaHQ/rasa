import argparse
from typing import Text

from rasa.cli.arguments.default_arguments import (
    add_data_param,
    add_domain_param,
    add_endpoint_param,
    add_nlu_data_param,
    add_out_param,
    add_sub_agents_param,
)
from rasa.shared.constants import DEFAULT_CONVERTED_DATA_PATH


def set_convert_nlu_arguments(parser: argparse.ArgumentParser, data_type: Text) -> None:
    """Sets convert nlu command arguments."""
    parser.add_argument(
        "-f",
        "--format",
        default="yaml",
        choices=["json", "yaml"],
        help="Output format the training data should be converted into.",
    )

    add_data_param(parser, required=True, data_type=data_type)

    add_out_param(
        parser,
        default=DEFAULT_CONVERTED_DATA_PATH,
        help_text="File (for `json`) or existing path (for `yaml`) "
        "where to save training data in Rasa format.",
    )

    parser.add_argument("-l", "--language", default="en", help="Language of data.")


def set_convert_e2e_arguments(parser: argparse.ArgumentParser) -> None:
    """Sets convert e2e command arguments.

    Args:
        parser: Parser we are going to attach arguments to.
    """
    parser.add_argument(
        "path",
        type=str,
        help="Path to the input CSV or XLS/XLSX file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="e2e_tests",
        help="Output directory to store the tests.",
    )
    parser.add_argument(
        "--sheet-name",
        type=str,
        help="Worksheet name containing relevant data. Mandatory for Excel file input.",
    )


def set_split_arguments(parser: argparse.ArgumentParser) -> None:
    add_nlu_data_param(parser, help_text="File or folder containing your NLU data.")

    parser.add_argument(
        "--training-fraction",
        type=float,
        default=0.8,
        help="Percentage of the data which should be in the training data.",
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Seed to generate the same train/test split.",
    )

    add_out_param(
        parser,
        default="train_test_split",
        help_text="Directory where the split files should be stored.",
    )


def set_validator_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--fail-on-warnings",
        default=False,
        action="store_true",
        help="Fail validation on warnings and errors. "
        "If omitted only errors will result in a non zero exit code.",
    )
    add_domain_param(parser)
    add_data_param(parser)
    add_sub_agents_param(parser)
    # Endpoints are optional for `data validate` command
    add_endpoint_param(
        parser,
        help_text="Configuration file for the connectors as a yml file.",
        default=None,
    )


def set_migrate_arguments(parser: argparse.ArgumentParser) -> None:
    """Sets migrate command arguments."""
    add_domain_param(parser)

    add_out_param(
        parser,
        default=None,
        help_text="Path (for `yaml`) where to save migrated domain in Rasa 3.0 format."
        "If none is specified, either a `new_domain.yml` file or `new_domain` folder "
        "will be created in the folder that contains the given domain.",
    )
