import argparse

from rasa.cli.arguments.default_arguments import (
    add_nlu_data_param,
    add_out_param,
    add_data_param,
    add_stories_param,
    add_domain_param,
)


def set_convert_arguments(parser: argparse.ArgumentParser):
    add_data_param(parser, required=True, default=None, data_type="Rasa NLU ")

    add_out_param(
        parser,
        required=True,
        default=None,
        help_text="File where to save training data in Rasa format.",
    )

    parser.add_argument("-l", "--language", default="en", help="Language of data.")

    parser.add_argument(
        "-f",
        "--format",
        required=True,
        choices=["json", "md"],
        help="Output format the training data should be converted into.",
    )


def set_split_arguments(parser: argparse.ArgumentParser):
    add_nlu_data_param(parser, help_text="File or folder containing your NLU data.")

    parser.add_argument(
        "--training-fraction",
        type=float,
        default=0.8,
        help="Percentage of the data which should be in the training data.",
    )

    add_out_param(
        parser,
        default="train_test_split",
        help_text="Directory where the split files should be stored.",
    )


def set_validator_arguments(parser: argparse.ArgumentParser):
    add_domain_param(parser)
    add_data_param(parser)
