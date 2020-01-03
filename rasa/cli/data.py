import logging
import argparse
import asyncio
import sys
from typing import List

from rasa import data
from rasa.cli.arguments import data as arguments
from rasa.cli.utils import get_validated_path
from rasa.constants import DEFAULT_DATA_PATH
from typing import NoReturn

logger = logging.getLogger(__name__)


# noinspection PyProtectedMember
def add_subparser(
    subparsers: argparse._SubParsersAction, parents: List[argparse.ArgumentParser]
):
    data_parser = subparsers.add_parser(
        "data",
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=parents,
        help="Utils for the Rasa training files.",
    )
    data_parser.set_defaults(func=lambda _: data_parser.print_help(None))

    data_subparsers = data_parser.add_subparsers()

    _add_data_convert_parsers(data_subparsers, parents)
    _add_data_split_parsers(data_subparsers, parents)
    _add_data_validate_parsers(data_subparsers, parents)


def _add_data_convert_parsers(data_subparsers, parents: List[argparse.ArgumentParser]):
    import rasa.nlu.convert as convert

    convert_parser = data_subparsers.add_parser(
        "convert",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=parents,
        help="Converts Rasa data between different formats.",
    )
    convert_parser.set_defaults(func=lambda _: convert_parser.print_help(None))

    convert_subparsers = convert_parser.add_subparsers()
    convert_nlu_parser = convert_subparsers.add_parser(
        "nlu",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=parents,
        help="Converts NLU data between Markdown and json formats.",
    )
    convert_nlu_parser.set_defaults(func=convert.main)

    arguments.set_convert_arguments(convert_nlu_parser)


def _add_data_split_parsers(data_subparsers, parents: List[argparse.ArgumentParser]):
    split_parser = data_subparsers.add_parser(
        "split",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=parents,
        help="Splits Rasa data into training and test data.",
    )
    split_parser.set_defaults(func=lambda _: split_parser.print_help(None))

    split_subparsers = split_parser.add_subparsers()
    nlu_split_parser = split_subparsers.add_parser(
        "nlu",
        parents=parents,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Performs a split of your NLU data into training and test data "
        "according to the specified percentages.",
    )
    nlu_split_parser.set_defaults(func=split_nlu_data)

    arguments.set_split_arguments(nlu_split_parser)


def _add_data_validate_parsers(data_subparsers, parents: List[argparse.ArgumentParser]):
    validate_parser = data_subparsers.add_parser(
        "validate",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=parents,
        help="Validates domain and data files to check for possible mistakes.",
    )
    validate_parser.add_argument(
        "--max-history",
        type=int,
        default=None,
        help="Assume this max_history setting for story structure validation.",
    )
    validate_parser.set_defaults(func=validate_files)
    arguments.set_validator_arguments(validate_parser)

    validate_subparsers = validate_parser.add_subparsers()
    story_structure_parser = validate_subparsers.add_parser(
        "stories",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=parents,
        help="Checks for inconsistencies in the story files.",
    )
    story_structure_parser.add_argument(
        "--max-history",
        type=int,
        help="Assume this max_history setting for validation.",
    )
    story_structure_parser.add_argument(
        "--prompt",
        action="store_true",
        default=False,
        help="Ask how conflicts should be fixed",
    )
    story_structure_parser.set_defaults(func=validate_stories)
    arguments.set_validator_arguments(story_structure_parser)


def split_nlu_data(args) -> None:
    from rasa.nlu.training_data.loading import load_data
    from rasa.nlu.training_data.util import get_file_format

    data_path = get_validated_path(args.nlu, "nlu", DEFAULT_DATA_PATH)
    data_path = data.get_nlu_directory(data_path)

    nlu_data = load_data(data_path)
    fformat = get_file_format(data_path)

    train, test = nlu_data.train_test_split(args.training_fraction, args.random_seed)

    train.persist(args.out, filename=f"training_data.{fformat}")
    test.persist(args.out, filename=f"test_data.{fformat}")


def validate_files(args) -> NoReturn:
    """Validate all files needed for training a model.

    Fails with a non-zero exit code if there are any errors in the data."""
    from rasa.core.validator import Validator
    from rasa.importers.rasa import RasaFileImporter

    loop = asyncio.get_event_loop()
    file_importer = RasaFileImporter(
        domain_path=args.domain, training_data_paths=args.data
    )

    validator = loop.run_until_complete(Validator.from_importer(file_importer))
    domain_is_valid = validator.verify_domain_validity()
    if not domain_is_valid:
        sys.exit(1)

    everything_is_alright = validator.verify_all(not args.fail_on_warnings)
    if not args.max_history:
        logger.info(
            "Will not test for inconsistencies in stories since "
            "you did not provide --max-history."
        )
    if args.max_history:
        # Only run story structure validation if everything else is fine
        # since this might take a while
        everything_is_alright = validator.verify_story_structure(
            not args.fail_on_warnings, max_history=args.max_history
        )
    sys.exit(0) if everything_is_alright else sys.exit(1)


def validate_stories(args):
    """Validate all files needed for training a model.

        Fails with a non-zero exit code if there are any errors in the data."""
    from rasa.core.validator import Validator
    from rasa.importers.rasa import RasaFileImporter

    # Check if a valid setting for `max_history` was given
    if not isinstance(args.max_history, int) or args.max_history < 1:
        logger.error("You have to provide a positive integer for --max-history.")
        sys.exit(1)

    # Prepare story and domain file import
    loop = asyncio.get_event_loop()
    file_importer = RasaFileImporter(
        domain_path=args.domain, training_data_paths=args.data
    )

    # Loads the stories
    validator = loop.run_until_complete(Validator.from_importer(file_importer))

    # If names are unique, look for story conflicts
    everything_is_alright = validator.verify_story_structure(
        not args.fail_on_warnings, max_history=args.max_history
    )

    if not everything_is_alright:
        print_error("Story validation completed with errors.")
        sys.exit(1)
