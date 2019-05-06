import os
import sys
from typing import Any, Callable, Dict, Optional, Text, List

from rasa.constants import DEFAULT_MODELS_PATH


def get_validated_path(
    current: Optional[Text],
    parameter: Text,
    default: Optional[Text] = None,
    none_is_valid: bool = False,
) -> Optional[Text]:
    """Check whether a file path or its default value is valid and returns it.

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
            reason_str = "'{}' not found.".format(current)
            if current is None:
                reason_str = "Parameter '{}' not set.".format(parameter)

            print_warning(
                "{} Using default location '{}' instead." "".format(reason_str, default)
            )
            current = default
        elif none_is_valid:
            current = None
        else:
            cancel_cause_not_found(current, parameter, default)

    return current


def missing_config_keys(path: Text, mandatory_keys: List[Text]) -> List:
    import rasa.utils.io

    if not os.path.exists(path):
        return mandatory_keys

    config_data = rasa.utils.io.read_yaml_file(path)

    return [k for k in mandatory_keys if k not in config_data or config_data[k] is None]


def cancel_cause_not_found(
    current: Optional[Text], parameter: Text, default: Optional[Text]
) -> None:
    """Exits with an error because the given path was not valid.

    Args:
        current: The path given by the user.
        parameter: The name of the parameter.
        default: The default value of the parameter.

    """

    default_clause = ""
    if default:
        default_clause = "use the default location ('{}') or ".format(default)
    print_error(
        "The path '{}' does not exist. Please make sure to {}specify it"
        " with '--{}'.".format(current, default_clause, parameter)
    )
    exit(1)


def parse_last_positional_argument_as_model_path() -> None:
    """Fixes the parsing of a potential positional model path argument."""
    import sys

    if (
        len(sys.argv) >= 2
        and sys.argv[1] in ["run", "shell", "interactive"]
        and not sys.argv[-2].startswith("-")
        and os.path.exists(sys.argv[-1])
    ):
        sys.argv.append(sys.argv[-1])
        sys.argv[-2] = "--model"


def create_output_path(
    output_path: Text = DEFAULT_MODELS_PATH, prefix: Text = ""
) -> Text:
    """Creates an output path which includes the current timestamp.

    Args:
        output_path: The path where the model should be stored.
        prefix: A prefix which should be included in the output path.

    Returns:
        The generated output path, e.g. "20191201-103002.tar.gz".
    """
    import time

    if output_path.endswith("tar.gz"):
        return output_path
    else:
        time_format = "%Y%m%d-%H%M%S"
        file_name = "{}{}.tar.gz".format(prefix, time.strftime(time_format))
        return os.path.join(output_path, file_name)


def minimal_kwargs(
    kwargs: Dict[Text, Any], func: Callable, excluded_keys: Optional[List] = None
) -> Dict[Text, Any]:
    """Returns only the kwargs which are required by a function. Keys, contained in
    the exception list, are not included.

    Args:
        kwargs: All available kwargs.
        func: The function which should be called.
        excluded_keys: Keys to exclude from the result.

    Returns:
        Subset of kwargs which are accepted by `func`.

    """
    from rasa.utils.common import arguments_of

    excluded_keys = excluded_keys or []

    possible_arguments = arguments_of(func)

    return {
        k: v
        for k, v in kwargs.items()
        if k in possible_arguments and k not in excluded_keys
    }


def print_success(*args: Any):
    print_color(*args, color=bcolors.OKGREEN)


class bcolors(object):
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def wrap_with_color(*args: Any, color: Text):
    return color + " ".join(str(s) for s in args) + bcolors.ENDC


def print_color(*args: Any, color: Text):
    print (wrap_with_color(*args, color=color))


def print_warning(*args: Any):
    print_color(*args, color=bcolors.WARNING)


def print_error(*args: Any):
    print_color(*args, color=bcolors.FAIL)


def signal_handler(sig, frame):
    print ("Goodbye 👋")
    sys.exit(0)
