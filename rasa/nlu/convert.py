import argparse

from rasa.nlu import training_data
from rasa.nlu.utils import write_to_file


def add_arguments(parser):
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
    return parser


def convert_training_data(data_file, out_file, output_format, language):
    td = training_data.load_data(data_file, language)

    if output_format == "md":
        output = td.as_markdown()
    else:
        output = td.as_json(indent=2)

    write_to_file(out_file, output)


def main(args):
    convert_training_data(args.data_file, args.out_file, args.format, args.language)


if __name__ == "__main__":
    raise RuntimeError(
        "Calling `rasa.nlu.convert` directly is "
        "no longer supported. "
        "Please use `rasa data` instead."
    )
