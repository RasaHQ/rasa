import importlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from types import FrameType
from typing import TYPE_CHECKING, Any, Dict, List, Text

import structlog

import rasa.shared.utils.io

if TYPE_CHECKING:
    from questionary import Question

structlogger = structlog.get_logger()

FREE_TEXT_INPUT_PROMPT = "Type out your own message..."


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
    if results_output_path.is_dir() or not results_output_path.suffix:
        file_name = results_output_path / f"e2e_results_{result_type}.yml"
    else:
        parent = results_output_path.parent
        stem = results_output_path.stem
        file_name = parent / f"{stem}_{result_type}.yml"

    return str(file_name)


def is_skip_validation_flag_set() -> bool:
    """Checks if the skip validation flag is set."""
    return "--skip-validation" in sys.argv


def get_failed_e2e_tests_file_name(
    failed_tests_path: Path,
) -> str:
    """Returns the name of the e2e failed tests file with timestamp.

    Args:
        failed_tests_path: Path provided by the user via CLI for failed tests output.

    Returns:
        Path to the failed tests file with timestamp as a string.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if failed_tests_path.is_dir() or not failed_tests_path.suffix:
        file_name = failed_tests_path / f"e2e_failed_tests_{timestamp}.yml"
    else:
        parent = failed_tests_path.parent
        stem = failed_tests_path.stem
        suffix = failed_tests_path.suffix
        file_name = parent / f"{stem}_{timestamp}{suffix}"

    return str(file_name)
