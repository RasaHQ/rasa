import json
import argparse
import structlog
import importlib
import os
import sys
import time
from pathlib import Path
from types import FrameType
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Text, Union, overload
import randomname

import rasa.shared.utils.cli
import rasa.shared.utils.io
from rasa.shared.importers.importer import TrainingDataImporter
from rasa.shared.constants import (
    ASSISTANT_ID_DEFAULT_VALUE,
    ASSISTANT_ID_KEY,
    DEFAULT_CONFIG_PATH,
)
from rasa import telemetry
from rasa.shared.utils.yaml import read_config_file
from rasa.utils.io import write_yaml

if TYPE_CHECKING:
    from questionary import Question
    from typing_extensions import Literal
    from rasa.validator import Validator

structlogger = structlog.get_logger()

FREE_TEXT_INPUT_PROMPT = "Type out your own message..."


@overload
def get_validated_path(
    current: Optional[Union[Path, Text]],
    parameter: Text,
    default: Optional[Union[Path, Text, List[Text]]] = ...,
    none_is_valid: "Literal[False]" = ...,
) -> Union[Path, Text]: ...


@overload
def get_validated_path(
    current: Optional[Union[Path, Text]],
    parameter: Text,
    default: Optional[Union[Path, Text, List[Text]]] = ...,
    none_is_valid: "Literal[True]" = ...,
) -> Optional[Union[Path, Text]]: ...


def get_validated_path(
    current: Optional[Union[Path, Text]],
    parameter: Text,
    default: Optional[Union[Path, Text, List[Text]]] = None,
    none_is_valid: bool = False,
) -> Optional[Union[Path, Text]]:
    """Checks whether a file path or its default value is valid and returns it.

    Args:
        current: The parsed value.
        parameter: The name of the parameter.
        default: one or multiple default values of the parameter.
        none_is_valid: `True` if `None` is valid value for the path,
                        else `False``

    Returns:
        The current value if valid,
        otherwise one of the default values of the argument if valid,
        otherwise `None` if allowed,
        otherwise raises an error and exits.
    """
    if current and os.path.exists(current):
        return current

    # try to find a valid option among the defaults
    if isinstance(default, str) or isinstance(default, Path):
        default_options = [str(default)]
    elif isinstance(default, list):
        default_options = default
    else:
        default_options = []

    valid_options = (option for option in default_options if os.path.exists(option))
    chosen_option = next(valid_options, None)

    # warn and log if user-chosen parameter wasn't found and thus overwritten
    if chosen_option:
        shared_info = f"Using default location '{chosen_option}' instead."
        if current is None:
            structlogger.debug(
                "cli.get_validated_path.parameter_not_set",
                parameter=parameter,
                event_info=(f"Parameter '{parameter}' was not set. {shared_info}"),
            )
        else:
            structlogger.warn(
                "cli.get_validated_path.path_does_not_exists",
                path=current,
                event_info=(
                    f"The path '{current}' does not seem to exist. {shared_info}"
                ),
            )

    if chosen_option is None and not none_is_valid:
        cancel_cause_not_found(current, parameter, default)

    return chosen_option


def missing_config_keys(
    path: Union["Path", Text], mandatory_keys: List[Text]
) -> List[Text]:
    """Checks whether the config file at `path` contains the `mandatory_keys`.

    Args:
        path: The path to the config file.
        mandatory_keys: A list of mandatory config keys.

    Returns:
        The list of missing config keys.
    """
    if not os.path.exists(path):
        return mandatory_keys

    config_data = read_config_file(path)

    return [k for k in mandatory_keys if k not in config_data or config_data[k] is None]


def validate_assistant_id_in_config(config_file: Union["Path", Text]) -> None:
    """Verifies that the assistant_id key exists and has a unique value in config.

    Issues a warning if the key does not exist or has the default value and replaces it
    with a pseudo-random string value.
    """
    config_data = read_config_file(config_file, reader_type=["safe", "rt"])
    assistant_id = config_data.get(ASSISTANT_ID_KEY)

    if assistant_id is None or assistant_id == ASSISTANT_ID_DEFAULT_VALUE:
        structlogger.warn(
            "cli.validate_assistant_id_in_config.missing_unique_assistant_id_key",
            config=config_file,
            missing_key=ASSISTANT_ID_KEY,
            event_info=(
                f"The config file '{config_file!s}' is "
                f"missing a unique value for the "
                f"'{ASSISTANT_ID_KEY}' mandatory key. "
                f"Proceeding with generating a random "
                f"value and overwriting the '{ASSISTANT_ID_KEY}'"
                f" in the config file."
            ),
        )

        # add random value for assistant id, overwrite config file
        time_format = "%Y%m%d-%H%M%S"
        config_data[ASSISTANT_ID_KEY] = (
            f"{time.strftime(time_format)}-{randomname.get_name()}"
        )

        write_yaml(data=config_data, target=config_file, should_preserve_key_order=True)

    return


def validate_config_path(
    config: Optional[Union[Text, "Path"]],
    default_config: Text = DEFAULT_CONFIG_PATH,
) -> Text:
    """Verifies that the config path exists.

    Exit if the config file does not exist.

    Args:
        config: Path to the config file.
        default_config: default config to use if the file at `config` doesn't exist.

    Returns: The path to the config file.
    """
    config = rasa.cli.utils.get_validated_path(config, "config", default_config)

    if not config or not os.path.exists(config):
        structlogger.error(
            "cli.validate_config_path.does_not_exists",
            config=config,
            event_info=(
                f"The config file '{config}' does not exist. "
                f"Use '--config' to specify a valid config file."
            ),
        )
        sys.exit(1)

    return str(config)


def validate_mandatory_config_keys(
    config: Union[Text, "Path"],
    mandatory_keys: List[Text],
) -> Text:
    """Get a config from a config file and check if it is valid.

    Exit if the config isn't valid.

    Args:
        config: Path to the config file.
        mandatory_keys: The keys that have to be specified in the config file.

    Returns: The path to the config file if the config is valid.
    """
    missing_keys = set(rasa.cli.utils.missing_config_keys(config, mandatory_keys))
    if missing_keys:
        structlogger.error(
            "cli.validate_mandatory_config_keys.missing_keys",
            config=config,
            missing_keys=missing_keys,
            event_info=(
                "The config file '{}' is missing mandatory parameters: "
                "'{}'. Add missing parameters to config file and try again.".format(
                    config, "', '".join(missing_keys)
                )
            ),
        )
        sys.exit(1)

    return str(config)


def get_validated_config(
    config: Optional[Union[Text, "Path"]],
    mandatory_keys: List[Text],
    default_config: Text = DEFAULT_CONFIG_PATH,
) -> Text:
    """Validates config and returns path to validated config file."""
    config = validate_config_path(config, default_config)
    validate_assistant_id_in_config(config)

    config = validate_mandatory_config_keys(config, mandatory_keys)

    return config


def validate_files(
    fail_on_warnings: bool,
    max_history: Optional[int],
    importer: TrainingDataImporter,
    stories_only: bool = False,
    flows_only: bool = False,
) -> None:
    """Validates either the story structure or the entire project.

    Args:
        fail_on_warnings: `True` if the process should exit with a non-zero status
        max_history: The max history to use when validating the story structure.
        importer: The `TrainingDataImporter` to use to load the training data.
        stories_only: If `True`, only the story structure is validated.
        flows_only: If `True`, only the flows are validated.
    """
    from rasa.validator import Validator

    validator = Validator.from_importer(importer)

    if stories_only:
        all_good = _validate_story_structure(validator, max_history, fail_on_warnings)
    elif flows_only:
        all_good = validator.verify_flows()
    else:
        if importer.get_domain().is_empty():
            structlogger.error(
                "cli.validate_files.empty_domain",
                event_info="Encountered empty domain during validation.",
            )
            sys.exit(1)

        valid_domain = _validate_domain(validator)
        valid_nlu = _validate_nlu(validator, fail_on_warnings)
        valid_stories = _validate_story_structure(
            validator, max_history, fail_on_warnings
        )
        valid_flows = validator.verify_flows()
        valid_CALM_slot_mappings = validator.validate_CALM_slot_mappings()

        all_good = (
            valid_domain
            and valid_nlu
            and valid_stories
            and valid_flows
            and valid_CALM_slot_mappings
        )

    validator.warn_if_config_mandatory_keys_are_not_set()

    telemetry.track_validate_files(all_good)
    if not all_good:
        structlogger.error(
            "cli.validate_files.project_validation_error",
            event_info="Project validation completed with errors.",
        )
        sys.exit(1)


def _validate_domain(validator: "Validator") -> bool:
    valid_domain_validity = validator.verify_domain_validity()
    valid_actions_in_stories_rules = validator.verify_actions_in_stories_rules()
    valid_forms_in_stories_rules = validator.verify_forms_in_stories_rules()
    valid_form_slots = validator.verify_form_slots()
    valid_slot_mappings = validator.verify_slot_mappings()
    valid_responses = validator.check_for_no_empty_paranthesis_in_responses()
    valid_buttons = validator.validate_button_payloads()
    return (
        valid_domain_validity
        and valid_actions_in_stories_rules
        and valid_forms_in_stories_rules
        and valid_form_slots
        and valid_slot_mappings
        and valid_responses
        and valid_buttons
    )


def _validate_nlu(validator: "Validator", fail_on_warnings: bool) -> bool:
    return validator.verify_nlu(not fail_on_warnings)


def _validate_story_structure(
    validator: "Validator", max_history: Optional[int], fail_on_warnings: bool
) -> bool:
    # Check if a valid setting for `max_history` was given
    if isinstance(max_history, int) and max_history < 1:
        raise argparse.ArgumentTypeError(
            f"The value of `--max-history {max_history}` " f"is not a positive integer."
        )

    return validator.verify_story_structure(
        not fail_on_warnings, max_history=max_history
    )


def cancel_cause_not_found(
    current: Optional[Union["Path", Text]],
    parameter: Text,
    default: Optional[Union["Path", Text, List[Text]]],
) -> None:
    """Exits with an error because the given path was not valid.

    Args:
        current: The path given by the user.
        parameter: The name of the parameter.
        default: The default value of the parameter.

    """
    default_clause = ""
    if default and isinstance(default, str):
        default_clause = f"use the default location ('{default}') or"
    elif default and isinstance(default, list):
        default_clause = f"use one of the default locations ({', '.join(default)}) or"

    structlogger.error(
        "cli.path_does_not_exist",
        path=current,
        event_info=(
            f"The path '{current}' does not exist. "
            f"Please make sure to {default_clause} specify it "
            f"with '--{parameter}'."
        ),
    )
    sys.exit(1)


def parse_last_positional_argument_as_model_path() -> None:
    """Fixes the parsing of a potential positional model path argument."""
    if (
        len(sys.argv) >= 2
        # support relevant commands ...
        and sys.argv[1] in ["run", "shell", "interactive"]
        # but avoid interpreting subparser commands as model paths
        and sys.argv[1:] != ["run", "actions"]
        and not sys.argv[-2].startswith("-")
        and os.path.exists(sys.argv[-1])
    ):
        sys.argv.append(sys.argv[-1])
        sys.argv[-2] = "--model"


def button_to_string(button: Dict[Text, Any], idx: int = 0) -> Text:
    """Create a string representation of a button."""
    title = button.pop("title", "")

    if "payload" in button:
        payload = " ({})".format(button.pop("payload"))
    else:
        payload = ""

    # if there are any additional attributes, we append them to the output
    if button:
        details = " - {}".format(json.dumps(button, sort_keys=True))
    else:
        details = ""

    button_string = "{idx}: {title}{payload}{details}".format(
        idx=idx + 1, title=title, payload=payload, details=details
    )

    return button_string


def element_to_string(element: Dict[Text, Any], idx: int = 0) -> Text:
    """Create a string representation of an element."""
    title = element.pop("title", "")

    element_string = "{idx}: {title} - {element}".format(
        idx=idx + 1, title=title, element=json.dumps(element, sort_keys=True)
    )

    return element_string


def button_choices_from_message_data(
    message: Dict[Text, Any], allow_free_text_input: bool = True
) -> List[Text]:
    """Return list of choices to present to the user.

    If allow_free_text_input is True, an additional option is added
    at the end along with the response buttons that allows the user
    to type in free text.
    """
    choices = [
        button_to_string(button, idx)
        for idx, button in enumerate(message.get("buttons"))
    ]
    if allow_free_text_input:
        choices.append(FREE_TEXT_INPUT_PROMPT)
    return choices


async def payload_from_button_question(button_question: "Question") -> Text:
    """Prompt user with a button question and returns the nlu payload."""
    response = await button_question.ask_async()
    if response != FREE_TEXT_INPUT_PROMPT:
        # Extract intent slash command if it's a button
        response = response[response.rfind("(") + 1 : response.rfind(")")]
    return response


def signal_handler(_: int, __: FrameType) -> None:
    """Kills Rasa when OS signal is received."""
    print("Goodbye 👋")
    sys.exit(0)


def warn_if_rasa_plus_package_installed() -> None:
    """Issue a user warning in case the `rasa_plus` package is installed."""
    rasa_plus_package = "rasa_plus"
    if importlib.util.find_spec(rasa_plus_package) is not None:
        rasa.shared.utils.io.raise_warning(
            f"{rasa_plus_package} python package is no longer necessary "
            f"for using Rasa Pro. Please uninstall it.",
            UserWarning,
        )


def check_if_studio_command() -> bool:
    """Checks if the command is a Rasa Studio command."""
    return len(sys.argv) >= 2 and sys.argv[1] == "studio"


def get_e2e_results_file_name(
    results_output_path: Path,
    result_type: str,
) -> str:
    """Returns the name of the e2e results file."""
    if results_output_path.is_dir():
        file_name = str(results_output_path) + f"/e2e_results_{result_type}.yml"
    else:
        parent = results_output_path.parent
        stem = results_output_path.stem
        file_name = str(parent) + f"/{stem}_{result_type}.yml"

    return file_name
