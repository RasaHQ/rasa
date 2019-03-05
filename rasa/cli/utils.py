import argparse
import os
from typing import Text, Optional, Dict, Callable, Any

from rasa.constants import DEFAULT_MODELS_PATH


def check_path_exists(current: Optional[Text], parameter: Text,
                      default: Optional[Text] = None,
                      none_is_valid: bool = False) -> Optional[Text]:
    """Check whether a file path which was given through the command line
    arguments is valid.

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
    from rasa_core.utils import print_warning

    if (current is None or
            current is not None and not os.path.exists(current)):
        if default is not None and os.path.exists(default):
            print_warning("'{}' not found. Using default location '{}' instead."
                          "".format(current, default))
            current = default
        elif none_is_valid:
            current = None
        else:
            cancel_cause_not_found(current, parameter, default)

    return current


def cancel_cause_not_found(current: Optional[Text], parameter: Text,
                           default: Optional[Text]) -> None:
    """Exits with an error because the given path was not valid.

    Args:
        current: The path given by the user.
        parameter: The name of the parameter.
        default: The default value of the parameter.

    """
    from rasa_core.utils import print_error

    default_clause = ""
    if default:
        default_clause = ("use the default location ('{}') or "
                          "".format(default))
    print_error("The path '{}' does not exist. Please make sure to {}specify it"
                " with '--{}'.".format(current, default_clause, parameter))
    exit(1)


def validate_path(args: argparse.Namespace,
                  name: Text, default: Optional[Text], is_none_allowed: bool
                  = False) -> None:
    """Validates the parsed command line argument whether its value or its
    default value is a valid path.

    Args:
        args: The parsed command line arguments.
        name: Name of the parameter to validate.
        default: Default value for this parameter.
        is_none_allowed: `True` if `None` is a valid value for this parameter.
    """
    validated = check_path_exists(getattr(args, name), name, default,
                                  is_none_allowed)
    setattr(args, name, validated)


def parse_last_positional_argument_as_model_path() -> None:
    """Fixes the parsing of a potential positional model path argument."""
    import sys

    if (len(sys.argv) >= 2 and
            sys.argv[1] in ["run", "test", "shell", "interactive"] and not
            sys.argv[-2].startswith('-') and
            os.path.exists(sys.argv[-1])):
        sys.argv.append(sys.argv[-1])
        sys.argv[-2] = "--model"


def create_output_path(output_path: Text = DEFAULT_MODELS_PATH,
                       prefix: Text = "") -> Text:
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


def minimal_kwargs(kwargs: Dict[Text, Any], func: Callable) -> Dict[Text, Any]:
    """Returns only the kwargs which are required by a function.

    Args:
        kwargs: All available kwargs.
        func: The function which should be called.

    Returns:
        Subset of kwargs which are accepted by `func`.

    """
    from rasa_core.utils import arguments_of

    possible_arguments = arguments_of(func)

    return {k: v for k, v in kwargs.items() if k in possible_arguments}
