import argparse
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, Union, List, Text, TYPE_CHECKING

import rasa.shared.core.domain
from rasa import telemetry
from rasa.cli import SubParsersAction
from rasa.cli.arguments import data as arguments
from rasa.cli.arguments import default_arguments
import rasa.cli.utils
from rasa.shared.constants import (
    DEFAULT_DATA_PATH,
    DEFAULT_CONFIG_PATH,
    DOCS_URL_MIGRATION_GUIDE,
)
import rasa.shared.data
from rasa.shared.core.constants import (
    POLICY_NAME_FALLBACK,
    POLICY_NAME_FORM,
    POLICY_NAME_MAPPING,
    POLICY_NAME_TWO_STAGE_FALLBACK,
    USER_INTENT_OUT_OF_SCOPE,
    ACTION_DEFAULT_FALLBACK_NAME,
)
from rasa.shared.core.training_data.story_reader.yaml_story_reader import (
    YAMLStoryReader,
)
from rasa.shared.core.training_data.story_writer.yaml_story_writer import (
    YAMLStoryWriter,
)
from rasa.shared.importers.rasa import RasaFileImporter
import rasa.shared.nlu.training_data.loading
import rasa.shared.nlu.training_data.util
import rasa.shared.utils.cli
import rasa.utils.common
from rasa.shared.core.domain import Domain, InvalidDomain
import rasa.shared.utils.io

if TYPE_CHECKING:
    from rasa.shared.core.training_data.structures import StoryStep
    from rasa.validator import Validator
    from rasa.utils.converter import TrainingDataConverter

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

    convert_nlg_parser = convert_subparsers.add_parser(
        "nlg",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=parents,
        help=(
            "Converts NLG data between formats. If you're migrating from 1.x, "
            "please run `rasa data convert responses` to adapt the training data "
            "to the new response selector format."
        ),
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
    default_arguments.add_config_param(migrate_config_parser)
    default_arguments.add_domain_param(migrate_config_parser)
    default_arguments.add_out_param(
        migrate_config_parser,
        default=os.path.join(DEFAULT_DATA_PATH, "rules.yml"),
        help_text="Path to the file which should contain any rules which are created "
        "as part of the migration. If the file doesn't exist, it will be created.",
    )

    convert_responses_parser = convert_subparsers.add_parser(
        "responses",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=parents,
        help=(
            "Convert retrieval intent responses between Rasa Open Source versions. "
            "Please also run `rasa data convert nlg` to convert training data files "
            "to the right format."
        ),
    )
    convert_responses_parser.set_defaults(func=_migrate_responses)
    arguments.set_convert_arguments(convert_responses_parser, data_type="Rasa stories")
    default_arguments.add_domain_param(convert_responses_parser)


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


def validate_files(args: argparse.Namespace, stories_only: bool = False) -> None:
    """Validates either the story structure or the entire project.

    Args:
        args: Commandline arguments
        stories_only: If `True`, only the story structure is validated.
    """
    from rasa.validator import Validator

    config = rasa.cli.utils.get_validated_path(
        args.config, "config", DEFAULT_CONFIG_PATH, none_is_valid=True
    )

    file_importer = RasaFileImporter(
        domain_path=args.domain, training_data_paths=args.data, config_file=config,
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


def _validate_domain(validator: "Validator") -> bool:
    return (
        validator.verify_domain_validity()
        and validator.verify_actions_in_stories_rules()
        and validator.verify_form_slots()
    )


def _validate_nlu(validator: "Validator", args: argparse.Namespace) -> bool:
    return validator.verify_nlu(not args.fail_on_warnings)


def _validate_story_structure(validator: "Validator", args: argparse.Namespace) -> bool:
    # Check if a valid setting for `max_history` was given
    if isinstance(args.max_history, int) and args.max_history < 1:
        raise argparse.ArgumentTypeError(
            f"The value of `--max-history {args.max_history}` "
            f"is not a positive integer."
        )

    return validator.verify_story_structure(
        not args.fail_on_warnings, max_history=args.max_history
    )


def _convert_nlu_data(args: argparse.Namespace) -> None:
    import rasa.nlu.convert

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
            _convert_to_yaml(args.out, args.data, NLUMarkdownToYamlConverter())
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
            _convert_to_yaml(args.out, args.data, StoryMarkdownToYamlConverter())
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
            _convert_to_yaml(args.out, args.data, NLGMarkdownToYamlConverter())
        )
        telemetry.track_data_convert(args.format, "nlg")
    else:
        rasa.shared.utils.cli.print_error_and_exit(
            "Could not recognize output format. Supported output formats: "
            "'yaml'. Specify the desired output format with '--format'."
        )


def _migrate_responses(args: argparse.Namespace) -> None:
    """Migrate retrieval intent responses to the new 2.0 format.

    It does so modifying the stories and domain files.
    """
    from rasa.core.training.converters.responses_prefix_converter import (
        DomainResponsePrefixConverter,
        StoryResponsePrefixConverter,
    )

    if args.format == "yaml":
        rasa.utils.common.run_in_loop(
            _convert_to_yaml(args.out, args.domain, DomainResponsePrefixConverter())
        )
        rasa.utils.common.run_in_loop(
            _convert_to_yaml(args.out, args.data, StoryResponsePrefixConverter())
        )
        telemetry.track_data_convert(args.format, "responses")
    else:
        rasa.shared.utils.cli.print_error_and_exit(
            "Could not recognize output format. Supported output formats: "
            "'yaml'. Specify the desired output format with '--format'."
        )


async def _convert_to_yaml(
    out_path: Text, data_path: Union[list, Text], converter: "TrainingDataConverter"
) -> None:

    if isinstance(data_path, list):
        data_path = data_path[0]

    output = Path(out_path)
    if not os.path.exists(output):
        rasa.shared.utils.cli.print_error_and_exit(
            f"The output path '{output}' doesn't exist. Please make sure to specify "
            f"an existing directory and try again."
        )

    training_data = Path(data_path)
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
    source_file: Path, target_dir: Path, converter: "TrainingDataConverter"
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


def _migrate_model_config(args: argparse.Namespace) -> None:
    """Migrates old "rule-like" policies to the new `RulePolicy`.

    Updates the config, domain, and generates the required rules.

    Args:
        args: The commandline args with the required paths.
    """
    import rasa.core.config

    configuration_file = Path(args.config)
    model_configuration = _get_configuration(configuration_file)

    domain_file = Path(args.domain)
    domain = _get_domain(domain_file)

    rule_output_file = _get_rules_path(args.out)

    (
        model_configuration,
        domain,
        new_rules,
    ) = rasa.core.config.migrate_mapping_policy_to_rules(model_configuration, domain)

    model_configuration, fallback_rule = rasa.core.config.migrate_fallback_policies(
        model_configuration
    )

    if new_rules:
        _backup(domain_file)
        domain.persist_clean(domain_file)

    if fallback_rule:
        new_rules.append(fallback_rule)

    if new_rules or model_configuration["policies"]:
        _backup(configuration_file)
        rasa.shared.utils.io.write_yaml(model_configuration, configuration_file)
        _dump_rules(rule_output_file, new_rules)

    telemetry.track_data_convert("yaml", "config")

    _print_success_message(new_rules, rule_output_file)


def _get_configuration(path: Path) -> Dict:
    config = {}
    try:
        config = rasa.shared.utils.io.read_model_configuration(path)
    except Exception:
        rasa.shared.utils.cli.print_error_and_exit(
            f"'{path}' is not a path to a valid model configuration. "
            f"Please provide a valid path."
        )

    policy_names = [p.get("name") for p in config.get("policies", [])]

    _assert_config_needs_migration(policy_names)
    _assert_nlu_pipeline_given(config, policy_names)
    _assert_two_stage_fallback_policy_is_migratable(config)
    _assert_only_one_fallback_policy_present(policy_names)

    if POLICY_NAME_FORM in policy_names:
        _warn_about_manual_forms_migration()

    return config


def _assert_config_needs_migration(policies: List[Text]) -> None:
    migratable_policies = {
        POLICY_NAME_MAPPING,
        POLICY_NAME_FALLBACK,
        POLICY_NAME_TWO_STAGE_FALLBACK,
    }

    if not migratable_policies.intersection((set(policies))):
        rasa.shared.utils.cli.print_error_and_exit(
            f"No policies were found which need migration. This command can migrate "
            f"'{POLICY_NAME_MAPPING}', '{POLICY_NAME_FALLBACK}' and "
            f"'{POLICY_NAME_TWO_STAGE_FALLBACK}'."
        )


def _warn_about_manual_forms_migration() -> None:
    rasa.shared.utils.cli.print_warning(
        f"Your model configuration contains the '{POLICY_NAME_FORM}'. "
        f"Note that this command does not migrate the '{POLICY_NAME_FORM}' and "
        f"you have to migrate the '{POLICY_NAME_FORM}' manually. "
        f"Please see the migration guide for further details: "
        f"{DOCS_URL_MIGRATION_GUIDE}"
    )


def _assert_nlu_pipeline_given(config: Dict, policy_names: List[Text]) -> None:
    if not config.get("pipeline") and any(
        policy in policy_names
        for policy in [POLICY_NAME_FALLBACK, POLICY_NAME_TWO_STAGE_FALLBACK]
    ):
        rasa.shared.utils.cli.print_error_and_exit(
            "The model configuration has to include an NLU pipeline. This is required "
            "in order to migrate the fallback policies."
        )


def _assert_two_stage_fallback_policy_is_migratable(config: Dict) -> None:
    two_stage_fallback_config = next(
        (
            policy_config
            for policy_config in config.get("policies", [])
            if policy_config.get("name") == POLICY_NAME_TWO_STAGE_FALLBACK
        ),
        None,
    )
    if not two_stage_fallback_config:
        return

    if (
        two_stage_fallback_config.get(
            "deny_suggestion_intent_name", USER_INTENT_OUT_OF_SCOPE
        )
        != USER_INTENT_OUT_OF_SCOPE
    ):
        rasa.shared.utils.cli.print_error_and_exit(
            f"The TwoStageFallback in Rasa Open Source 2.0 has to use the intent "
            f"'{USER_INTENT_OUT_OF_SCOPE}' to recognize when users deny suggestions. "
            f"Please change the parameter 'deny_suggestion_intent_name' to "
            f"'{USER_INTENT_OUT_OF_SCOPE}' before migrating the model configuration. "
        )

    if (
        two_stage_fallback_config.get(
            "fallback_nlu_action_name", ACTION_DEFAULT_FALLBACK_NAME
        )
        != ACTION_DEFAULT_FALLBACK_NAME
    ):
        rasa.shared.utils.cli.print_error_and_exit(
            f"The Two-Stage Fallback in Rasa Open Source 2.0 has to use the action "
            f"'{ACTION_DEFAULT_FALLBACK_NAME}' for cases when the user denies the "
            f"suggestion multiple times. "
            f"Please change the parameter 'fallback_nlu_action_name' to "
            f"'{ACTION_DEFAULT_FALLBACK_NAME}' before migrating the model "
            f"configuration. "
        )


def _assert_only_one_fallback_policy_present(policies: List[Text]) -> None:
    if POLICY_NAME_FALLBACK in policies and POLICY_NAME_TWO_STAGE_FALLBACK in policies:
        rasa.shared.utils.cli.print_error_and_exit(
            "Your policy configuration contains two configured policies for handling "
            "fallbacks. Please decide on one."
        )


def _get_domain(path: Path) -> Domain:
    try:
        return Domain.from_path(path)
    except InvalidDomain:
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

    if not rules_file.is_file():
        rasa.shared.utils.cli.print_info(
            f"Output file '{rules_file}' did not exist and will be created."
        )
        rasa.shared.utils.io.create_directory_for_file(rules_file)

    return rules_file


def _dump_rules(path: Path, new_rules: List["StoryStep"]) -> None:
    existing_rules = []
    if path.exists():
        rules_reader = YAMLStoryReader()
        existing_rules = rules_reader.read_from_file(path)
        _backup(path)

    if existing_rules:
        rasa.shared.utils.cli.print_info(
            f"Found existing rules in the output file '{path}'. The new rules will "
            f"be appended to the existing rules."
        )

    rules_writer = YAMLStoryWriter()
    rules_writer.dump(path, existing_rules + new_rules)


def _backup(path: Path) -> None:
    backup_file = path.parent / f"{path.name}.bak"
    shutil.copy(path, backup_file)


def _print_success_message(new_rules: List["StoryStep"], output_file: Path) -> None:
    if len(new_rules) > 1 or len(new_rules) == 0:
        rules_text = "rules"
        verb = "were"
    else:
        rules_text = "rule"
        verb = "was"

    rasa.shared.utils.cli.print_success(
        "Finished migrating your policy configuration 🎉."
    )
    if len(new_rules) == 0:
        rasa.shared.utils.cli.print_success(
            f"The migration generated {len(new_rules)} {rules_text} so no {rules_text} "
            f"{verb} added to '{output_file}'."
        )
    else:
        rasa.shared.utils.cli.print_success(
            f"The migration generated {len(new_rules)} {rules_text} which {verb} added "
            f"to '{output_file}'."
        )
