import json
import logging
import os
import sys
import time
from types import FrameType
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Text, Union, overload

import randomname

import rasa.shared.utils.cli
import rasa.shared.utils.io
from rasa.shared.constants import (
    ASSISTANT_ID_DEFAULT_VALUE,
    ASSISTANT_ID_KEY,
    DEFAULT_CONFIG_PATH,
)
from rasa.shared.utils.cli import print_error

if TYPE_CHECKING:
    from pathlib import Path

    from questionary import Question
    from typing_extensions import Literal

logger = logging.getLogger(__name__)

FREE_TEXT_INPUT_PROMPT = "Type out your own message..."


@overload
def get_validated_path(
    current: Optional[Union["Path", Text]],
    parameter: Text,
    default: Optional[Union["Path", Text]] = ...,
    none_is_valid: "Literal[False]" = ...,
) -> Union["Path", Text]:
    ...


@overload
def get_validated_path(
    current: Optional[Union["Path", Text]],
    parameter: Text,
    default: Optional[Union["Path", Text]] = ...,
    none_is_valid: "Literal[True]" = ...,
) -> Optional[Union["Path", Text]]:
    ...


def get_validated_path(
    current: Optional[Union["Path", Text]],
    parameter: Text,
    default: Optional[Union["Path", Text]] = None,
    none_is_valid: bool = False,
) -> Optional[Union["Path", Text]]:
    """Checks whether a file path or its default value is valid and returns it.

    Args:
        current: The parsed value.
        parameter: The name of the parameter.
        default: The default value of the parameter.
        none_is_valid: `True` if `None` is valid value for the path,
                        else `False``

    Returns:
        The current value if it was valid, else the default value of the
        argument if it is valid, else `None`.
    """
    if current is None or current is not None and not os.path.exists(current):
        if default is not None and os.path.exists(default):
            reason_str = f"'{current}' not found."
            if current is None:
                reason_str = f"Parameter '{parameter}' not set."
            else:
                rasa.shared.utils.io.raise_warning(
                    f"The path '{current}' does not seem to exist. Using the "
                    f"default value '{default}' instead."
                )

            logger.debug(f"{reason_str} Using default location '{default}' instead.")
            current = default
        elif none_is_valid:
            current = None
        else:
            cancel_cause_not_found(current, parameter, default)

    return current


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
    import rasa.utils.io

    if not os.path.exists(path):
        return mandatory_keys

    config_data = rasa.shared.utils.io.read_config_file(path)

    return [k for k in mandatory_keys if k not in config_data or config_data[k] is None]


def validate_assistant_id_in_config(config_file: Union["Path", Text]) -> None:
    """Verifies that the assistant_id key exists and has a unique value in config.

    Issues a warning if the key does not exist or has the default value and replaces it
    with a pseudo-random string value.
    """
    config_data = rasa.shared.utils.io.read_config_file(config_file)
    assistant_id = config_data.get(ASSISTANT_ID_KEY)

    if assistant_id is None or assistant_id == ASSISTANT_ID_DEFAULT_VALUE:
        rasa.shared.utils.io.raise_warning(
            f"The config file '{str(config_file)}' is missing a unique value for the "
            f"'{ASSISTANT_ID_KEY}' mandatory key. Proceeding with generating a random "
            f"value and overwriting the '{ASSISTANT_ID_KEY}' in the config file."
        )

        # add random value for assistant id, overwrite config file
        time_format = "%Y%m%d-%H%M%S"
        config_data[
            ASSISTANT_ID_KEY
        ] = f"{time.strftime(time_format)}-{randomname.get_name()}"

        rasa.shared.utils.io.write_yaml(
            data=config_data, target=config_file, should_preserve_key_order=True
        )

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
        print_error(
            "The config file '{}' does not exist. Use '--config' to specify a "
            "valid config file."
            "".format(config)
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
        print_error(
            "The config file '{}' is missing mandatory parameters: "
            "'{}'. Add missing parameters to config file and try again."
            "".format(config, "', '".join(missing_keys))
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


def cancel_cause_not_found(
    current: Optional[Union["Path", Text]],
    parameter: Text,
    default: Optional[Union["Path", Text]],
) -> None:
    """Exits with an error because the given path was not valid.

    Args:
        current: The path given by the user.
        parameter: The name of the parameter.
        default: The default value of the parameter.

    """
    default_clause = ""
    if default:
        default_clause = f"use the default location ('{default}') or "
    rasa.shared.utils.cli.print_error(
        "The path '{}' does not exist. Please make sure to {}specify it"
        " with '--{}'.".format(current, default_clause, parameter)
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
        response = response[response.find("(") + 1 : response.find(")")]
    return response


def signal_handler(_: int, __: FrameType) -> None:
    """Kills Rasa when OS signal is received."""
    print("Goodbye 👋")
    sys.exit(0)
