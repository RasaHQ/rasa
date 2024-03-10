import argparse
import logging
import pathlib
from typing import List

import rasa.shared.core.domain
from rasa import telemetry
from rasa.cli import SubParsersAction
from rasa.cli.arguments import data as arguments
from rasa.cli.arguments import default_arguments
import rasa.cli.utils
from rasa.shared.constants import (
    DEFAULT_DATA_PATH,
    DEFAULT_CONFIG_PATH,
    DEFAULT_DOMAIN_PATH,
)
import rasa.shared.data
from rasa.shared.importers.importer import TrainingDataImporter
import rasa.shared.nlu.training_data.loading
import rasa.shared.nlu.training_data.util
import rasa.shared.utils.cli
import rasa.utils.common
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
    _add_data_migrate_parsers(data_subparsers, parents)


def _add_data_convert_parsers(
    data_subparsers: SubParsersAction, parents: List[argparse.ArgumentParser]
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


def _add_data_split_parsers(
    data_subparsers: SubParsersAction, parents: List[argparse.ArgumentParser]
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

    stories_split_parser = split_subparsers.add_parser(
        "stories",
        parents=parents,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Performs a split of your stories into training and test data "
        "according to the specified percentages.",
    )
    stories_split_parser.set_defaults(func=split_stories_data)

    arguments.set_split_arguments(stories_split_parser)


def _add_data_validate_parsers(
    data_subparsers: SubParsersAction, parents: List[argparse.ArgumentParser]
) -> None:
    validate_parser = data_subparsers.add_parser(
        "validate",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=parents,
        help="Validates domain and data files to check for possible mistakes.",
    )
    _append_story_structure_arguments(validate_parser)
    validate_parser.set_defaults(
        func=lambda args: rasa.cli.utils.validate_files(
            args.fail_on_warnings, args.max_history, _build_training_data_importer(args)
        )
    )
    arguments.set_validator_arguments(validate_parser)

    validate_subparsers = validate_parser.add_subparsers()
    story_structure_parser = validate_subparsers.add_parser(
        "stories",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=parents,
        help="Checks for inconsistencies in the story files.",
    )
    _append_story_structure_arguments(story_structure_parser)

    story_structure_parser.set_defaults(
        func=lambda args: rasa.cli.utils.validate_files(
            args.fail_on_warnings,
            args.max_history,
            _build_training_data_importer(args),
            stories_only=True,
        )
    )
    arguments.set_validator_arguments(story_structure_parser)


def _build_training_data_importer(args: argparse.Namespace) -> "TrainingDataImporter":
    config = rasa.cli.utils.get_validated_path(
        args.config, "config", DEFAULT_CONFIG_PATH, none_is_valid=True
    )

    # Exit the validation if the domain path is invalid
    domain = rasa.cli.utils.get_validated_path(
        args.domain, "domain", DEFAULT_DOMAIN_PATH, none_is_valid=False
    )

    return TrainingDataImporter.load_from_config(
        domain_path=domain, training_data_paths=args.data, config_path=config
    )


def _append_story_structure_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--max-history",
        type=int,
        default=None,
        help="Number of turns taken into account for story structure validation.",
    )
    default_arguments.add_config_param(parser)


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


def split_stories_data(args: argparse.Namespace) -> None:
    """Load data from a file path and split stories into test and train examples.

    Args:
        args: Commandline arguments
    """
    from rasa.shared.core.training_data.story_reader.yaml_story_reader import (
        YAMLStoryReader,
        KEY_STORIES,
    )
    from sklearn.model_selection import train_test_split

    data_path = rasa.cli.utils.get_validated_path(args.nlu, "nlu", DEFAULT_DATA_PATH)
    data_files = rasa.shared.data.get_data_files(
        data_path, YAMLStoryReader.is_stories_file
    )
    out_path = pathlib.Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)

    # load Yaml stories data
    for file_name in data_files:
        file_data = rasa.shared.utils.io.read_yaml_file(file_name)
        assert isinstance(file_data, dict)
        stories = file_data.get(KEY_STORIES, [])
        if not stories:
            logger.info(f"File {file_name} has no stories, skipped")
            continue

        file_path = pathlib.Path(file_name)

        # everything besides stories are going into the training data
        train, test = train_test_split(
            stories, test_size=1 - args.training_fraction, random_state=args.random_seed
        )
        out_file_train = out_path / ("train_" + file_path.name)
        out_file_test = out_path / ("test_" + file_path.name)

        # train file contains everything else from the file + train stories
        file_data[KEY_STORIES] = train
        rasa.shared.utils.io.write_yaml(file_data, out_file_train)

        # test file contains just test stories
        rasa.shared.utils.io.write_yaml({KEY_STORIES: test}, out_file_test)
        logger.info(
            f"From {file_name} we produced {out_file_train} "
            f"with {len(train)} stories and {out_file_test} "
            f"with {len(test)} stories"
        )


def _convert_nlu_data(args: argparse.Namespace) -> None:
    import rasa.nlu.convert

    if args.format in ["json", "yaml"]:
        rasa.nlu.convert.convert_training_data(
            args.data, args.out, args.format, args.language
        )
        telemetry.track_data_convert(args.format, "nlu")
    else:
        rasa.shared.utils.cli.print_error_and_exit(
            "Could not recognize output format. Supported output formats: 'json' "
            "and 'yaml'. Specify the desired output format with '--format'."
        )


def _add_data_migrate_parsers(
    data_subparsers: SubParsersAction, parents: List[argparse.ArgumentParser]
) -> None:
    migrate_parser = data_subparsers.add_parser(
        "migrate",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=parents,
        help="Converts Rasa domain 2.0 format to required format for 3.0.",
    )
    migrate_parser.set_defaults(func=_migrate_domain)

    arguments.set_migrate_arguments(migrate_parser)


def _migrate_domain(args: argparse.Namespace) -> None:
    import rasa.core.migrate

    rasa.core.migrate.migrate_domain_format(args.domain, args.out)
