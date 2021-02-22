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


def set_augment_arguments(parser: argparse.ArgumentParser):
    add_config_param(parser)

    parser.add_argument(
        "--nlu-training-data",
        type=str,
        help="File containing your NLU training data, i.e. the train set generated with the `rasa data split nlu` command.",
    )

    parser.add_argument(
        "--nlu-evaluation-data",
        type=str,
        help="File containing your NLU evaluation data, i.e. the test set generated with the `rasa data split nlu` command.",
    )

    parser.add_argument(
        "--nlu-classification-report",
        default="intent_report.json",
        type=str,
        help="File containing your NLU classification report.",
    )

    parser.add_argument(
        "--paraphrases",
        type=str,
        help="File containing your paraphrases generated from the paraphraser repository: https://github.com/RasaHQ/rasa/pull/7584.",
    )

    parser.add_argument(
        "--intent-proportion",
        type=float,
        default=0.5,
        help="The proportion of intents (out of all intents) considered for data augmentation. The actual number of intents considered for data augmentation is determined on the basis of several factors, such as their current performance statistics or the number of available training examples.",
    )

    parser.add_argument(
        "--min-paraphrase-sim-score",
        type=float,
        default=0.8,
        help="Minimum similarity score threshold for paraphrases, i.e. any paraphrase with a score < min-paraphrase-sim-score "
        "will be discarded.",
    )

    parser.add_argument(
        "--max-paraphrase-sim-score",
        type=float,
        default=0.98,
        help="Maximum similarity threshold for paraphrases, i.e. any paraphrases with a score > max-paraphrase-sim-score "
             "will be discarded."
    )

    add_out_param(
        parser,
        default="nlu_augment",
        help_text="Directory where the augmented training data and the reports from model runs with data augmentation "
        "should be stored.",
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=29306,
        help="Seed to generate a random sample of paraphrases.",
    )
