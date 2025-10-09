import argparse
import logging
import pathlib
from typing import List, Optional

import rasa.cli.utils
import rasa.shared.core.domain
import rasa.shared.data
import rasa.shared.nlu.training_data.loading
import rasa.shared.nlu.training_data.util
import rasa.shared.utils.cli
import rasa.shared.utils.io
import rasa.utils.common
from rasa import telemetry
from rasa.cli import SubParsersAction
from rasa.cli.arguments import data as arguments
from rasa.cli.arguments import default_arguments
from rasa.cli.validation.bot_config import validate_files
from rasa.cli.validation.config_path_validation import get_validated_path
from rasa.core.config.configuration import Configuration
from rasa.e2e_test.e2e_config import create_llm_e2e_test_converter_config
from rasa.e2e_test.e2e_test_converter import E2ETestConverter
from rasa.e2e_test.utils.e2e_yaml_utils import E2ETestYAMLWriter
from rasa.shared.constants import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_DATA_PATH,
    DEFAULT_DOMAIN_PATHS,
)
from rasa.shared.exceptions import RasaException
from rasa.shared.importers.importer import TrainingDataImporter
from rasa.shared.utils.common import minimal_kwargs
from rasa.shared.utils.yaml import read_yaml_file, write_yaml
from rasa.utils.beta import ensure_beta_feature_is_enabled

logger = logging.getLogger(__name__)

RASA_PRO_BETA_E2E_CONVERSION_ENV_VAR_NAME = "RASA_PRO_BETA_E2E_CONVERSION"


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
    arguments.set_convert_nlu_arguments(convert_nlu_parser, data_type="Rasa NLU")

    convert_e2e_parser = convert_subparsers.add_parser(
        "e2e",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=parents,
        help="Convert input sample conversations into E2E test cases.",
    )
    convert_e2e_parser.set_defaults(func=convert_data_to_e2e_tests)
    arguments.set_convert_e2e_arguments(convert_e2e_parser)


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
        func=lambda args: _validate_files_with_configuration(
            args.fail_on_warnings,
            args.max_history,
            _build_training_data_importer(args),
            sub_agents=args.sub_agents,
            endpoints=args.endpoints,
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
        func=lambda args: _validate_files_with_configuration(
            args.fail_on_warnings,
            args.max_history,
            _build_training_data_importer(args),
            sub_agents=args.sub_agents,
            endpoints=args.endpoints,
            stories_only=True,
        )
    )
    arguments.set_validator_arguments(story_structure_parser)

    flows_structure_parser = validate_subparsers.add_parser(
        "flows",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=parents,
        help="Checks for inconsistencies in the flows files.",
    )
    flows_structure_parser.set_defaults(
        func=lambda args: _validate_files_with_configuration(
            args.fail_on_warnings,
            args.max_history,
            _build_training_data_importer(args),
            sub_agents=args.sub_agents,
            endpoints=args.endpoints,
            flows_only=True,
        )
    )
    arguments.set_validator_arguments(flows_structure_parser)

    translations_structure_parser = validate_subparsers.add_parser(
        "translations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=parents,
        help="Checks for inconsistencies of the flow and response translation.",
    )
    translations_structure_parser.set_defaults(
        func=lambda args: _validate_files_with_configuration(
            args.fail_on_warnings,
            args.max_history,
            _build_training_data_importer(args),
            sub_agents=args.sub_agents,
            endpoints=args.endpoints,
            translations_only=True,
        )
    )
    arguments.set_validator_arguments(translations_structure_parser)


def _build_training_data_importer(args: argparse.Namespace) -> "TrainingDataImporter":
    config = get_validated_path(
        args.config, "config", DEFAULT_CONFIG_PATH, none_is_valid=True
    )

    # Exit the validation if the domain path is invalid
    domain = get_validated_path(
        args.domain, "domain", DEFAULT_DOMAIN_PATHS, none_is_valid=False
    )

    if config:
        return TrainingDataImporter.load_from_config(
            domain_path=domain, training_data_paths=args.data, config_path=config
        )
    else:
        return TrainingDataImporter.load_from_dict(
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
    data_path = get_validated_path(args.nlu, "nlu", DEFAULT_DATA_PATH)
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
    from sklearn.model_selection import train_test_split

    from rasa.shared.core.training_data.story_reader.yaml_story_reader import (
        KEY_STORIES,
        YAMLStoryReader,
    )

    data_path = get_validated_path(args.nlu, "nlu", DEFAULT_DATA_PATH)
    data_files = rasa.shared.data.get_data_files(
        data_path, YAMLStoryReader.is_stories_file
    )
    out_path = pathlib.Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)

    # load Yaml stories data
    for file_name in data_files:
        file_data = read_yaml_file(file_name)
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
        write_yaml(file_data, out_file_train)

        # test file contains just test stories
        write_yaml({KEY_STORIES: test}, out_file_test)
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


def validate_e2e_test_conversion_output_path(output_path: str) -> None:
    """Validates that the provided output path is within the project directory.

    Args:
        output_path (str): The output path to be validated.

    Raises:
        RasaException: If the provided output path is an absolute path.
    """
    if pathlib.Path(output_path).is_absolute():
        raise RasaException(
            "Please provide a relative output path within the assistant "
            "project directory in which the command is running."
        )


def convert_data_to_e2e_tests(args: argparse.Namespace) -> None:
    """Converts sample conversation data into E2E test cases
    and stores them in the output YAML file.

    Args:
        args: The arguments passed in from the CLI.
    """
    try:
        ensure_beta_feature_is_enabled(
            "conversion of sample conversations into end-to-end tests",
            RASA_PRO_BETA_E2E_CONVERSION_ENV_VAR_NAME,
        )
        validate_e2e_test_conversion_output_path(args.output)

        config_path = pathlib.Path(args.output)
        llm_config = create_llm_e2e_test_converter_config(config_path)

        kwargs = minimal_kwargs(vars(args), E2ETestConverter)
        converter = E2ETestConverter(llm_config=llm_config, **kwargs)
        yaml_tests_string = converter.run()

        writer = E2ETestYAMLWriter(output_path=args.output)
        writer.write_to_file(yaml_tests_string)
    except RasaException as exc:
        rasa.shared.utils.cli.print_error_and_exit(
            f"Failed to convert the data into E2E tests. Error: {exc}"
        )


def _initialize_configuration_for_validation(
    sub_agents: Optional[str] = None, endpoints: Optional[str] = None
) -> None:
    """Initialize Configuration before validation.

    Args:
        sub_agents: Path to sub-agents directory for validation.
        endpoints: Path to the endpoints configuration file.
    """
    if endpoints:
        Configuration.initialise_endpoints(endpoints_path=pathlib.Path(endpoints))
    if sub_agents:
        Configuration.initialise_sub_agents(pathlib.Path(sub_agents))
    if not endpoints and not sub_agents:
        Configuration.initialise_empty()


def _validate_files_with_configuration(
    fail_on_warnings: bool,
    max_history: int,
    importer: TrainingDataImporter,
    sub_agents: Optional[str] = None,
    endpoints: Optional[str] = None,
    stories_only: bool = False,
    flows_only: bool = False,
    translations_only: bool = False,
) -> None:
    """Wrapper for validate_files that ensures Configuration is initialized first.

    Args:
        fail_on_warnings: `True` if the process should exit with a non-zero status
        max_history: The max history to use when validating the story structure.
        importer: The `TrainingDataImporter` to use to load the training data.
        sub_agents: Path to sub-agents directory for validation.
        endpoints: Path to the endpoints configuration file.
        stories_only: If `True`, only the story structure is validated.
        flows_only: If `True`, only the flows are validated.
        translations_only: If `True`, only the translations data is validated.
    """
    # Initialize Configuration before calling validate_files
    _initialize_configuration_for_validation(sub_agents=sub_agents, endpoints=endpoints)

    # Call the original validate_files function
    validate_files(
        fail_on_warnings=fail_on_warnings,
        max_history=max_history,
        importer=importer,
        stories_only=stories_only,
        flows_only=flows_only,
        translations_only=translations_only,
        sub_agents=sub_agents,
        endpoints=endpoints,
    )
