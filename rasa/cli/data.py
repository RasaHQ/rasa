import argparse
import asyncio
import logging
import os
from pathlib import Path
from typing import List

import rasa.shared.data
from rasa.cli.arguments import data as arguments
import rasa.cli.utils
from rasa.cli.utils import print_error_and_exit, print_info, print_warning
from rasa.constants import DEFAULT_DATA_PATH
from rasa.shared.data import is_valid_filetype
from rasa.importers.rasa import RasaFileImporter
from rasa.nlu.convert import convert_training_data
from rasa.utils.converter import TrainingDataConverter
from rasa.validator import Validator

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


def _add_data_convert_parsers(
    data_subparsers, parents: List[argparse.ArgumentParser]
) -> None:
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
        help="Converts NLU data between formats.",
    )
    convert_nlu_parser.set_defaults(func=_convert_nlu_data)

    arguments.set_convert_arguments(convert_nlu_parser, data_type="Rasa NLU")

    convert_nlg_parser = convert_subparsers.add_parser(
        "nlg",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=parents,
        help="Converts NLG data between formats.",
    )
    convert_nlg_parser.set_defaults(func=_convert_nlg_data)

    arguments.set_convert_arguments(convert_nlg_parser, data_type="Rasa NLG")

    convert_core_parser = convert_subparsers.add_parser(
        "core",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=parents,
        help="Converts Core data between formats.",
    )
    convert_core_parser.set_defaults(func=_convert_core_data)

    arguments.set_convert_arguments(convert_core_parser, data_type="Rasa Core")


def _add_data_split_parsers(
    data_subparsers, parents: List[argparse.ArgumentParser]
) -> None:
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
) -> None:
    validate_parser = data_subparsers.add_parser(
        "validate",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=parents,
        help="Validates domain and data files to check for possible mistakes.",
    )
    _append_story_structure_arguments(validate_parser)
    validate_parser.set_defaults(func=validate_files)
    arguments.set_validator_arguments(validate_parser)

    validate_subparsers = validate_parser.add_subparsers()
    story_structure_parser = validate_subparsers.add_parser(
        "stories",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=parents,
        help="Checks for inconsistencies in the story files.",
    )
    _append_story_structure_arguments(story_structure_parser)
    story_structure_parser.set_defaults(func=validate_stories)
    arguments.set_validator_arguments(story_structure_parser)


def _append_story_structure_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--max-history",
        type=int,
        default=None,
        help="Number of turns taken into account for story structure validation.",
    )


def split_nlu_data(args: argparse.Namespace) -> None:
    """Load data from a file path and split the NLU data into test and train examples.

    Args:
        args: Commandline arguments
    """
    from rasa.shared.nlu.training_data.loading import load_data
    from rasa.shared.nlu.training_data.util import get_file_format

    data_path = rasa.cli.utils.get_validated_path(args.nlu, "nlu", DEFAULT_DATA_PATH)
    data_path = rasa.shared.data.get_nlu_directory(data_path)

    nlu_data = load_data(data_path)
    fformat = get_file_format(data_path)

    train, test = nlu_data.train_test_split(args.training_fraction, args.random_seed)

    train.persist(args.out, filename=f"training_data.{fformat}")
    test.persist(args.out, filename=f"test_data.{fformat}")


def validate_files(args: argparse.Namespace, stories_only: bool = False) -> None:
    """Validates either the story structure or the entire project.

    Args:
        args: Commandline arguments
        stories_only: If `True`, only the story structure is validated.
    """
    loop = asyncio.get_event_loop()
    file_importer = RasaFileImporter(
        domain_path=args.domain, training_data_paths=args.data
    )

    validator = loop.run_until_complete(Validator.from_importer(file_importer))

    if stories_only:
        all_good = _validate_story_structure(validator, args)
    else:
        all_good = (
            _validate_domain(validator)
            and _validate_nlu(validator, args)
            and _validate_story_structure(validator, args)
        )

    if not all_good:
        rasa.cli.utils.print_error_and_exit("Project validation completed with errors.")


def validate_stories(args: argparse.Namespace) -> None:
    """Validates that training data file content conforms to training data spec.

    Args:
        args: Commandline arguments
    """
    validate_files(args, stories_only=True)


def _validate_domain(validator: Validator) -> bool:
    return validator.verify_domain_validity()


def _validate_nlu(validator: Validator, args: argparse.Namespace) -> bool:
    return validator.verify_nlu(not args.fail_on_warnings)


def _validate_story_structure(validator: Validator, args: argparse.Namespace) -> bool:
    # Check if a valid setting for `max_history` was given
    if isinstance(args.max_history, int) and args.max_history < 1:
        raise argparse.ArgumentTypeError(
            f"The value of `--max-history {args.max_history}` is not a positive integer."
        )

    return validator.verify_story_structure(
        not args.fail_on_warnings, max_history=args.max_history
    )


def _convert_nlu_data(args: argparse.Namespace) -> None:
    from rasa.nlu.training_data.converters.nlu_markdown_to_yaml_converter import (
        NLUMarkdownToYamlConverter,
    )

    if args.format in ["json", "md"]:
        convert_training_data(args.data, args.out, args.format, args.language)
    elif args.format == "yaml":
        _convert_to_yaml(args, NLUMarkdownToYamlConverter())
    else:
        print_error_and_exit(
            "Could not recognize output format. Supported output formats: 'json', "
            "'md', 'yaml'. Specify the desired output format with '--format'."
        )


def _convert_core_data(args: argparse.Namespace) -> None:
    from rasa.core.training.converters.story_markdown_to_yaml_converter import (
        StoryMarkdownToYamlConverter,
    )

    if args.format == "yaml":
        _convert_to_yaml(args, StoryMarkdownToYamlConverter())
    else:
        print_error_and_exit(
            "Could not recognize output format. Supported output formats: "
            "'yaml'. Specify the desired output format with '--format'."
        )


def _convert_nlg_data(args: argparse.Namespace) -> None:
    from rasa.nlu.training_data.converters.nlg_markdown_to_yaml_converter import (
        NLGMarkdownToYamlConverter,
    )

    if args.format == "yaml":
        _convert_to_yaml(args, NLGMarkdownToYamlConverter())
    else:
        print_error_and_exit(
            "Could not recognize output format. Supported output formats: "
            "'yaml'. Specify the desired output format with '--format'."
        )


def _convert_to_yaml(
    args: argparse.Namespace, converter: TrainingDataConverter
) -> None:

    output = Path(args.out)
    if not os.path.exists(output):
        print_error_and_exit(
            f"The output path '{output}' doesn't exist. Please make sure to specify "
            f"an existing directory and try again."
        )

    training_data = Path(args.data)
    if not os.path.exists(training_data):
        print_error_and_exit(
            f"The training data path {training_data} doesn't exist "
            f"and will be skipped."
        )

    num_of_files_converted = 0

    if os.path.isfile(training_data):
        if _convert_file_to_yaml(training_data, output, converter):
            num_of_files_converted += 1
    elif os.path.isdir(training_data):
        for root, _, files in os.walk(training_data, followlinks=True):
            for f in sorted(files):
                source_path = Path(os.path.join(root, f))
                if _convert_file_to_yaml(source_path, output, converter):
                    num_of_files_converted += 1

    if num_of_files_converted:
        print_info(f"Converted {num_of_files_converted} file(s), saved in '{output}'.")
    else:
        print_warning(
            f"Didn't convert any files under '{training_data}' path. "
            "Did you specify the correct file/directory?"
        )


def _convert_file_to_yaml(
    source_file: Path, target_dir: Path, converter: TrainingDataConverter
) -> bool:
    """Converts a single training data file to `YAML` format.

    Args:
        source_file: Training data file to be converted.
        target_dir: Target directory for the converted file.
        converter: Converter to be used.

    Returns:
        `True` if file was converted, `False` otherwise.
    """
    if not is_valid_filetype(source_file):
        return False

    if converter.filter(source_file):
        converter.convert_and_write(source_file, target_dir)
        return True

    print_warning(f"Skipped file: '{source_file}'.")

    return False
