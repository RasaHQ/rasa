import logging
import argparse
import asyncio
import sys
from typing import List

from rasa import data
from rasa.cli.arguments import data as arguments
from rasa.cli.utils import get_validated_path
from rasa.constants import DEFAULT_DATA_PATH

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
    _add_data_clean_parsers(data_subparsers, parents)


def _add_data_convert_parsers(
        data_subparsers, parents: List[argparse.ArgumentParser]
):
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


def _add_data_split_parsers(
        data_subparsers, parents: List[argparse.ArgumentParser]
):
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


def _add_data_validate_parsers(
        data_subparsers, parents: List[argparse.ArgumentParser]
):
    validate_parser = data_subparsers.add_parser(
        "validate",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=parents,
        help="Validates domain and data files to check for possible mistakes.",
    )
    validate_parser.add_argument("--max-history", type=int, default=None,
                                 help="Assume this max_history setting for story structure validation.")
    validate_parser.set_defaults(func=validate_files)
    arguments.set_validator_arguments(validate_parser)

    validate_subparsers = validate_parser.add_subparsers()
    story_structure_parser = validate_subparsers.add_parser(
        "stories",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=parents,
        help="Checks for inconsistencies in the story files.",
    )
    story_structure_parser.add_argument("--max-history", type=int,
                                        help="Assume this max_history setting for validation.")
    story_structure_parser.add_argument("--prompt", action="store_true", default=False,
                                        help="Ask how conflicts should be fixed")
    story_structure_parser.set_defaults(func=validate_stories)
    arguments.set_validator_arguments(story_structure_parser)


def _add_data_clean_parsers(
    data_subparsers, parents: List[argparse.ArgumentParser]
):

    clean_parser = data_subparsers.add_parser(
        "clean",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=parents,
        help="[Experimental] Ensures that story names are unique.",
    )
    clean_parser.set_defaults(func=deduplicate_story_names)
    arguments.set_validator_arguments(clean_parser)


def split_nlu_data(args):
    from rasa.nlu.training_data.loading import load_data
    from rasa.nlu.training_data.util import get_file_format

    data_path = get_validated_path(args.nlu, "nlu", DEFAULT_DATA_PATH)
    data_path = data.get_nlu_directory(data_path)

    nlu_data = load_data(data_path)
    fformat = get_file_format(data_path)

    train, test = nlu_data.train_test_split(args.training_fraction)

    train.persist(args.out, filename=f"training_data.{fformat}")
    test.persist(args.out, filename=f"test_data.{fformat}")


def validate_files(args):
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
        logger.info("Will not test for inconsistencies in stories since "
                    "you did not provide --max-history.")
    if everything_is_alright and args.max_history:
        # Only run story structure validation if everything else is fine
        # since this might take a while
        everything_is_alright = validator.verify_story_structure(not args.fail_on_warnings,
                                                                 max_history=args.max_history)
    sys.exit(0) if everything_is_alright else sys.exit(1)


def validate_stories(args):
    """Validate all files needed for training a model.

        Fails with a non-zero exit code if there are any errors in the data."""
    from rasa.core.validator import Validator
    from rasa.importers.rasa import RasaFileImporter

    if not isinstance(args.max_history, int) or args.max_history < 1:
        logger.error("You have to provide a positive integer for --max-history.")
        sys.exit(1)

    loop = asyncio.get_event_loop()
    file_importer = RasaFileImporter(
        domain_path=args.domain, training_data_paths=args.data
    )

    # This loads the stories and thus fills `STORY_NAME_TALLY` (see next code block)
    validator = loop.run_until_complete(Validator.from_importer(file_importer))

    # First check for duplicate story names
    from rasa.core.training.structures import STORY_NAME_TALLY  # ToDo: Avoid global variable
    duplicate_story_names = {name: count for (name, count) in STORY_NAME_TALLY.items() if count > 1}
    story_names_unique = len(duplicate_story_names) == 0
    if not story_names_unique:
        msg = "Found duplicate story names:\n"
        for (name, count) in duplicate_story_names.items():
            msg += f"  '{name}' appears {count}x\n"
        logger.error(msg)

    # If names are unique, look for inconsistencies
    everything_is_alright = validator.verify_story_structure(
        not args.fail_on_warnings,
        max_history=args.max_history,
        prompt=args.prompt
    ) if story_names_unique else False

    sys.exit(0) if everything_is_alright else sys.exit(1)


def deduplicate_story_names(args):
    """Changes story names so as to make them unique.
       --EXPERIMENTAL-- """

    # ToDo: Make this work with multiple story files

    from rasa.importers.rasa import RasaFileImporter

    loop = asyncio.get_event_loop()
    file_importer = RasaFileImporter(
        domain_path=args.domain, training_data_paths=args.data
    )

    import shutil

    story_file_names, _ = data.get_core_nlu_files(args.data)
    names = set()
    for file_name in story_file_names:
        if file_name.endswith(".bak"):
            continue

        shutil.copy2(file_name, file_name + ".bak")

        with open(file_name, "r") as in_file, \
                open(file_name + ".new", "w+") as out_file:
            for line in in_file:
                line = line.rstrip()
                if line.startswith("## "):
                    new_name = line[3:]
                    if new_name in names:
                        first = new_name
                        k = 1
                        while new_name in names:
                            new_name = first + f" ({k})"
                            k += 1
                        print(f"- replacing {first} with {new_name}")
                    names.add(new_name)
                    out_file.write(f"## {new_name}\n")
                else:
                    out_file.write(line + "\n")

        shutil.move(file_name + ".new", file_name)

    # story_files, _ = data.get_core_nlu_files(args.data)
    # story_steps = loop.run_until_complete(file_importer.get_story_steps())
    # for step in story_steps:
    #     print(step.block_name)
