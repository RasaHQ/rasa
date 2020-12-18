import argparse
from typing import Text

from rasa.cli.arguments.default_arguments import (
    add_nlu_data_param,
    add_out_param,
    add_data_param,
    add_domain_param,
    add_config_param,
)
from rasa.shared.constants import DEFAULT_CONVERTED_DATA_PATH


def set_convert_arguments(parser: argparse.ArgumentParser, data_type: Text):
    parser.add_argument(
        "-f",
        "--format",
        default="yaml",
        choices=["json", "md", "yaml"],
        help="Output format the training data should be converted into. "
        "Note: currently training data can be converted to 'yaml' format "
        "only from 'md' format",
    )

    add_data_param(parser, required=True, data_type=data_type)

    add_out_param(
        parser,
        default=DEFAULT_CONVERTED_DATA_PATH,
        help_text="File (for `json` and `md`) or existing path (for `yaml`) "
        "where to save training data in Rasa format.",
    )

    parser.add_argument("-l", "--language", default="en", help="Language of data.")


def set_split_arguments(parser: argparse.ArgumentParser):
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


def set_validator_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--fail-on-warnings",
        default=False,
        action="store_true",
        help="Fail validation on warnings and errors. "
        "If omitted only errors will result in a non zero exit code.",
    )
    add_domain_param(parser)
    add_data_param(parser)


def set_suggest_arguments(parser: argparse.ArgumentParser):
    add_config_param(parser)

    parser.add_argument(
        "--nlu-training-data", type=str, help="File containing your NLU training data."
    )

    parser.add_argument(
        "--nlu-evaluation-data",
        type=str,
        help="File containing your NLU evaluation data.",
    )

    parser.add_argument(
        "--nlu-classification-report",
        type=str,
        help="File containing your NLU classification report.",
    )

    parser.add_argument(
        "--paraphrases", type=str, help="File containing your paraphrases."
    )

    parser.add_argument(
        "--num-intents",
        type=int,
        default=5,
        help="Number of intents per criterion for which to suggest paraphrases for.",
    )

    parser.add_argument(
        "--paraphrase-score-threshold",
        type=float,
        default=0.8,
        help="Quality score threshold for paraphrases, i.e. any paraphrase with a score < paraphrase-score-threshold "
        "will be discarded.",
    )

    add_out_param(
        parser,
        default="nlu_suggest",
        help_text="Directory where the augmented training data and the reports from model runs with data augmentation "
        "should be stored.",
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=29306,
        help="Seed to generate a random sample of paraphrases.",
    )
