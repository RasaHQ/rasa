import argparse
import logging
import os
from pathlib import Path
from typing import List, Text, Dict

import rasa.shared.core.domain
from rasa import telemetry
from rasa.cli import SubParsersAction
from rasa.cli.arguments import data as arguments
import rasa.cli.utils
import rasa.nlu.convert
from rasa.shared.constants import (
    DEFAULT_DATA_PATH,
    DEFAULT_CONFIG_PATH,
    DEFAULT_DOMAIN_PATH,
    DOCS_URL_MIGRATION_GUIDE,
)
import rasa.shared.data
from rasa.shared.importers.rasa import RasaFileImporter
import rasa.shared.nlu.training_data.loading
import rasa.shared.nlu.training_data.util
import rasa.shared.utils.cli
import rasa.utils.common
from rasa.utils.converter import TrainingDataConverter
from rasa.validator import Validator
from rasa.shared.core.domain import Domain, InvalidDomain
import rasa.shared.utils.io

logger = logging.getLogger(__name__)


def add_subparser(
    subparsers: SubParsersAction, parents: List[argparse.ArgumentParser]
) -> None:
    """Add all data parsers.

    Args:
        subparsers: subparser we are going to attach to
        parents: Parent parsers, needed to ensure tree structure in argparse
    """
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

    migrate_config_parser = convert_subparsers.add_parser(
        "config",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=parents,
        help="Migrate model configuration between Rasa Open Source versions.",
    )
    migrate_config_parser.set_defaults(func=_migrate_model_config)
    _add_migrate_config_arguments(migrate_config_parser)


def _add_migrate_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-c",
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Path to the model configuration which should be migrated",
    )
    parser.add_argument(
        "-d", "--domain", default=DEFAULT_DOMAIN_PATH, help="Path to the model domain"
    )
    parser.add_argument(
        "-o",
        "--out",
        type=str,
        default=os.path.join(DEFAULT_DATA_PATH, "rules.yml"),
        help="Path to the file which should contain any rules which are created as "
        "part of the migration. If the file doesn't exist, it will be created.",
    )


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
    data_path = rasa.cli.utils.get_validated_path(args.nlu, "nlu", DEFAULT_DATA_PATH)
    data_path = rasa.shared.data.get_nlu_directory(data_path)

    nlu_data = rasa.shared.nlu.training_data.loading.load_data(data_path)
    extension = rasa.shared.nlu.training_data.util.get_file_format_extension(data_path)

    train, test = nlu_data.train_test_split(args.training_fraction, args.random_seed)

    train.persist(args.out, filename=f"training_data{extension}")
    test.persist(args.out, filename=f"test_data{extension}")

    telemetry.track_data_split(args.training_fraction, "nlu")


def validate_files(args: argparse.Namespace, stories_only: bool = False) -> None:
    """Validates either the story structure or the entire project.

    Args:
        args: Commandline arguments
        stories_only: If `True`, only the story structure is validated.
    """
    file_importer = RasaFileImporter(
        domain_path=args.domain, training_data_paths=args.data
    )

    validator = rasa.utils.common.run_in_loop(Validator.from_importer(file_importer))

    if stories_only:
        all_good = _validate_story_structure(validator, args)
    else:
        all_good = (
            _validate_domain(validator)
            and _validate_nlu(validator, args)
            and _validate_story_structure(validator, args)
        )

    telemetry.track_validate_files(all_good)
    if not all_good:
        rasa.shared.utils.cli.print_error_and_exit(
            "Project validation completed with errors."
        )


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
        rasa.nlu.convert.convert_training_data(
            args.data, args.out, args.format, args.language
        )
        telemetry.track_data_convert(args.format, "nlu")
    elif args.format == "yaml":
        rasa.utils.common.run_in_loop(
            _convert_to_yaml(args, NLUMarkdownToYamlConverter())
        )
        telemetry.track_data_convert(args.format, "nlu")
    else:
        rasa.shared.utils.cli.print_error_and_exit(
            "Could not recognize output format. Supported output formats: 'json', "
            "'md', 'yaml'. Specify the desired output format with '--format'."
        )


def _convert_core_data(args: argparse.Namespace) -> None:
    from rasa.core.training.converters.story_markdown_to_yaml_converter import (
        StoryMarkdownToYamlConverter,
    )

    if args.format == "yaml":
        rasa.utils.common.run_in_loop(
            _convert_to_yaml(args, StoryMarkdownToYamlConverter())
        )
        telemetry.track_data_convert(args.format, "core")
    else:
        rasa.shared.utils.cli.print_error_and_exit(
            "Could not recognize output format. Supported output formats: "
            "'yaml'. Specify the desired output format with '--format'."
        )


def _convert_nlg_data(args: argparse.Namespace) -> None:
    from rasa.nlu.training_data.converters.nlg_markdown_to_yaml_converter import (
        NLGMarkdownToYamlConverter,
    )

    if args.format == "yaml":
        rasa.utils.common.run_in_loop(
            _convert_to_yaml(args, NLGMarkdownToYamlConverter())
        )
        telemetry.track_data_convert(args.format, "nlg")
    else:
        rasa.shared.utils.cli.print_error_and_exit(
            "Could not recognize output format. Supported output formats: "
            "'yaml'. Specify the desired output format with '--format'."
        )


async def _convert_to_yaml(
    args: argparse.Namespace, converter: TrainingDataConverter
) -> None:

    output = Path(args.out)
    if not os.path.exists(output):
        rasa.shared.utils.cli.print_error_and_exit(
            f"The output path '{output}' doesn't exist. Please make sure to specify "
            f"an existing directory and try again."
        )

    training_data = Path(args.data)
    if not os.path.exists(training_data):
        rasa.shared.utils.cli.print_error_and_exit(
            f"The training data path {training_data} doesn't exist "
            f"and will be skipped."
        )

    num_of_files_converted = 0

    if os.path.isfile(training_data):
        if await _convert_file_to_yaml(training_data, output, converter):
            num_of_files_converted += 1
    elif os.path.isdir(training_data):
        for root, _, files in os.walk(training_data, followlinks=True):
            for f in sorted(files):
                source_path = Path(os.path.join(root, f))
                if await _convert_file_to_yaml(source_path, output, converter):
                    num_of_files_converted += 1

    if num_of_files_converted:
        rasa.shared.utils.cli.print_info(
            f"Converted {num_of_files_converted} file(s), saved in '{output}'."
        )
    else:
        rasa.shared.utils.cli.print_warning(
            f"Didn't convert any files under '{training_data}' path. "
            "Did you specify the correct file/directory?"
        )


async def _convert_file_to_yaml(
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
    if not rasa.shared.data.is_valid_filetype(source_file):
        return False

    if converter.filter(source_file):
        await converter.convert_and_write(source_file, target_dir)
        return True

    rasa.shared.utils.cli.print_warning(f"Skipped file: '{source_file}'.")

    return False


def _migrate_model_config(args: argparse.Namespace):
    configuration_file = Path(args.config)
    model_configuration = _get_configuration(configuration_file)

    domain_file = Path(args.domain)
    domain = _get_domain(domain_file)

    rule_output_file = _get_rules_path(Path(args.out))

    # TODO:
    # 1. Migrate
    # 2. Dump config
    # 3. Dump rules
    # 4. Add telemetry


def _get_configuration(path: Path) -> Dict:
    from rasa.core.policies.form_policy import FormPolicy
    from rasa.core.policies.fallback import FallbackPolicy
    from rasa.core.policies.two_stage_fallback import TwoStageFallbackPolicy
    from rasa.core.policies.mapping_policy import MappingPolicy

    config = {}
    try:
        config = rasa.shared.utils.io.read_config_file(path)
    except ValueError:
        rasa.shared.utils.cli.print_error_and_exit(
            f"'{path}' is not a path to a valid model configuration. "
            f"Please provide a valid path."
        )

    policy_names = [p.get("name") for p in config.get("policies", [])]
    if not config.get("pipeline") and any(
        policy in policy_names
        for policy in [FallbackPolicy.__name__, TwoStageFallbackPolicy.__name__]
    ):
        rasa.shared.utils.cli.print_error_and_exit(
            f"The model configuration has to include an NLU pipeline. This is required "
            f"to migrate the fallback policies."
        )

    if FormPolicy.__name__ in policy_names:
        rasa.shared.utils.cli.print_error_and_exit(
            f"Your model configuration contains the '{FormPolicy.__name__}'. "
            f"Forms have to be migrated manually before '{MappingPolicy.__name__}', "
            f"'{FallbackPolicy.__name__}', or '{TwoStageFallbackPolicy.__name__}' "
            f"can be migrated. Please see the migration guide for further details: "
            f" {DOCS_URL_MIGRATION_GUIDE}"
        )

    return config


def _get_domain(path: Path) -> Domain:
    try:
        return Domain.from_path(path)
    except InvalidDomain as e:
        rasa.shared.utils.cli.print_error_and_exit(
            f"'{path}' is not a path to a valid domain file. "
            f"Please provide a valid domain."
        )


def _get_rules_path(path: Text) -> Path:
    rules_file = Path(path)

    if rules_file.is_dir():
        rasa.shared.utils.cli.print_error_and_exit(
            f"'{rules_file}' needs to be the path to a file."
        )

    return rules_file
