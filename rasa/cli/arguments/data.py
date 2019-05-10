import argparse

from rasa.cli.arguments.default_arguments import add_nlu_data_param


def set_convert_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-d", "--data_file", required=True, help="file or dir containing training data"
    )

    parser.add_argument(
        "-o",
        "--out_file",
        required=True,
        help="file where to save training data in rasa format",
    )

    parser.add_argument("-l", "--language", default="en", help="language of the data")

    parser.add_argument(
        "-f",
        "--format",
        required=True,
        choices=["json", "md"],
        help="Output format the training data should be converted into.",
    )


def set_split_arguments(parser: argparse.ArgumentParser):
    add_nlu_data_param(parser)

    parser.add_argument(
        "--training-fraction",
        type=float,
        default=0.8,
        help="Percentage of the data which should be the training data",
    )

    parser.add_argument(
        "-o",
        "--out",
        type=str,
        default="train_test_split",
        help="Directory where the split files should be stored",
    )
