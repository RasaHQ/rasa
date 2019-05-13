import argparse
from typing import List, Text

from rasa.nlu.training_data.loading import _guess_format
from rasa.nlu.utils import list_files

from rasa import data
from rasa.cli.arguments import data as arguments
from rasa.cli.utils import get_validated_path
from rasa.constants import DEFAULT_DATA_PATH


# noinspection PyProtectedMember
def add_subparser(
    subparsers: argparse._SubParsersAction, parents: List[argparse.ArgumentParser]
):
    import rasa.nlu.convert as convert

    data_parser = subparsers.add_parser(
        "data",
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=parents,
        help="Utils for the Rasa training files",
    )
    data_parser.set_defaults(func=lambda _: data_parser.print_help(None))

    data_subparsers = data_parser.add_subparsers()
    convert_parser = data_subparsers.add_parser(
        "convert",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=parents,
        help="Convert Rasa data between different formats",
    )
    convert_parser.set_defaults(func=lambda _: convert_parser.print_help(None))

    convert_subparsers = convert_parser.add_subparsers()
    convert_nlu_parser = convert_subparsers.add_parser(
        "nlu",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=parents,
        help="Convert NLU training data between markdown and json",
    )
    convert_nlu_parser.set_defaults(func=convert.main)

    arguments.set_convert_arguments(convert_nlu_parser)

    split_parser = data_subparsers.add_parser(
        "split",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=parents,
        help="Split Rasa data in training and test data",
    )
    split_parser.set_defaults(func=lambda _: split_parser.print_help(None))

    split_subparsers = split_parser.add_subparsers()
    nlu_split_parser = split_subparsers.add_parser(
        "nlu",
        parents=parents,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Perform a split of your NLU data according to the specified "
        "percentages",
    )
    nlu_split_parser.set_defaults(func=split_nlu_data)

    arguments.set_split_arguments(nlu_split_parser)


def split_nlu_data(args):
    from rasa.nlu.training_data.loading import load_data
    from rasa.nlu.training_data.util import get_file_format

    data_path = get_validated_path(args.nlu, "nlu", DEFAULT_DATA_PATH)
    data_path = data.get_nlu_directory(data_path)

    nlu_data = load_data(data_path)
    fformat = get_file_format(data_path)

    train, test = nlu_data.train_test_split(args.training_fraction)

    train.persist(
        args.out, filename="training_data.{}".format(fformat), fformat=fformat
    )
    test.persist(args.out, filename="test_data.{}".format(fformat), fformat=fformat)
