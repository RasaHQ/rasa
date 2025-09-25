import os
import time
from pathlib import Path
from typing import List, Optional, Text, Union

import randomname
import structlog

from rasa.exceptions import ModelNotFound, ValidationError
from rasa.shared.constants import (
    ASSISTANT_ID_DEFAULT_VALUE,
    ASSISTANT_ID_KEY,
    DEFAULT_CONFIG_PATH,
)
from rasa.shared.utils.common import display_research_study_prompt
from rasa.shared.utils.yaml import read_config_file
from rasa.utils.io import write_yaml

structlogger = structlog.get_logger()


FREE_TEXT_INPUT_PROMPT = "Type out your own message..."


def get_validated_path(
    current: Optional[Union[Path, Text]],
    parameter: Text,
    default: Optional[Union[Path, Text, List[Text], List[Path]]] = None,
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

    if parameter == "model":
        raise ModelNotFound(
            f"The provided model path '{current}' could not be found. "
            "Provide an existing model path."
        )

    default_options: Union[List[str], List[Path]] = []
    # try to find a valid option among the defaults
    if isinstance(default, str) or isinstance(default, Path):
        default_options = [str(default)]
    elif isinstance(default, list):
        default_options = default

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
        elif current not in default_options:
            structlogger.warn(
                "cli.get_validated_path.path_does_not_exists",
                path=current,
                event_info=(
                    f"The path '{current}' does not seem to exist. {shared_info}"
                ),
            )

    if chosen_option is None and not none_is_valid:
        _cancel_cause_not_found(current, parameter, default)

    return chosen_option


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
    config = get_validated_path(config, "config", default_config)

    if not config or not os.path.exists(config):
        display_research_study_prompt()
        raise ValidationError(
            code="cli.validate_config_path.does_not_exists",
            config=config,
            event_info=(
                f"The config file '{config}' does not exist. "
                f"Use '--config' to specify a valid config file."
            ),
        )

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
    missing_keys = set(_missing_config_keys(config, mandatory_keys))
    if missing_keys:
        display_research_study_prompt()
        raise ValidationError(
            code="cli.validate_mandatory_config_keys.missing_keys",
            config=config,
            missing_keys=missing_keys,
            event_info=(
                "The config file '{}' is missing mandatory parameters: "
                "'{}'. Add missing parameters to config file and try again.".format(
                    config, "', '".join(missing_keys)
                )
            ),
        )

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


def _missing_config_keys(
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


def _cancel_cause_not_found(
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
    display_research_study_prompt()
    raise ValidationError(
        code="cli.path_does_not_exist",
        path=current,
        event_info=(
            f"The path '{current}' does not exist. "
            f"Please make sure to {default_clause} specify it "
            f"with '--{parameter}'."
        ),
    )
