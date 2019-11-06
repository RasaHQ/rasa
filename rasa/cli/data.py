import argparse
import asyncio
import sys
from typing import List

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
        help="Utils for the Rasa training files.",
    )
    data_parser.set_defaults(func=lambda _: data_parser.print_help(None))

    data_subparsers = data_parser.add_subparsers()
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

    validate_parser = data_subparsers.add_parser(
        "validate",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=parents,
        help="Validates domain and data files to check for possible mistakes.",
    )
    validate_parser.add_argument("--stories", action="store_true", default=False,
                                 help="Also validate that stories are consistent.")
    validate_parser.add_argument("--max-history", type=int, default=5,
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
    story_structure_parser.add_argument("--max-history", type=int, default=5,
                                        help="Assume this max_history setting for validation.")
    story_structure_parser.set_defaults(func=validate_stories)
    arguments.set_validator_arguments(story_structure_parser)

    split_parser = data_subparsers.add_parser(
        "clean",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=parents,
        help="[Experimental] Ensures that story names are unique.",
    )
    split_parser.set_defaults(func=deduplicate_story_names)
    arguments.set_validator_arguments(split_parser)


def split_nlu_data(args):
    from rasa.nlu.training_data.loading import load_data
    from rasa.nlu.training_data.util import get_file_format

    data_path = get_validated_path(args.nlu, "nlu", DEFAULT_DATA_PATH)
    data_path = data.get_nlu_directory(data_path)

    nlu_data = load_data(data_path)
    fformat = get_file_format(data_path)

    train, test = nlu_data.train_test_split(args.training_fraction)

    train.persist(args.out, filename="training_data.{}".format(fformat))
    test.persist(args.out, filename="test_data.{}".format(fformat))


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
    everything_is_alright = validator.verify_all(not args.fail_on_warnings)
    if args.stories:
        everything_is_alright = everything_is_alright and \
                                validator.verify_story_structure(not args.fail_on_warnings,
                                                                 max_history=args.max_history)
    sys.exit(0) if everything_is_alright else sys.exit(1)


def validate_stories(args):
    """Validate all files needed for training a model.

        Fails with a non-zero exit code if there are any errors in the data."""
    from rasa.core.validator import Validator
    from rasa.importers.rasa import RasaFileImporter

    loop = asyncio.get_event_loop()
    file_importer = RasaFileImporter(
        domain_path=args.domain, training_data_paths=args.data
    )

    validator = loop.run_until_complete(Validator.from_importer(file_importer))
    everything_is_alright = (
            validator.verify_story_names(not args.fail_on_warnings) and
            validator.verify_story_structure(not args.fail_on_warnings, max_history=args.max_history)
    )
    sys.exit(0) if everything_is_alright else sys.exit(1)


def deduplicate_story_names(args):
    """Changes story names so as to make them unique.
       --EXPERIMENTAL-- """

    from rasa.importers.rasa import RasaFileImporter

    loop = asyncio.get_event_loop()
    file_importer = RasaFileImporter(
        domain_path=args.domain, training_data_paths=args.data
    )

    story_file_names, _ = data.get_core_nlu_files(args.data)
    names = set()
    for file_name in story_file_names:
        if file_name.endswith(".new"):
            continue
        with open(file_name, "r") as in_file:
            with open(file_name + ".new", "w+") as out_file:
                for line in in_file:
                    if line.startswith("## "):
                        new_name = line[3:].rstrip()
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
                        out_file.write(line.rstrip() + "\n")

    # story_files, _ = data.get_core_nlu_files(args.data)
    # story_steps = loop.run_until_complete(file_importer.get_story_steps())
    # for step in story_steps:
    #     print(step.block_name)
