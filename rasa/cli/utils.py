import argparse
import os
from typing import Text, Optional, List, Tuple, Union

from rasa.model import DEFAULT_MODELS_PATH


def check_path_exists(current: Optional[Text], parameter: Text,
                      default: Optional[Text] = None,
                      none_is_valid: bool = False) -> Optional[Text]:
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
    from rasa_core.utils import print_error

    default_clause = ""
    if default:
        default_clause = ("use the default location ('{}') or "
                          "".format(default))
    print_error("The path '{}' does not exist. Please make sure to {}specify it"
                " with '--{}'.".format(current, default_clause, parameter))
    exit(1)


def validate(args: argparse.Namespace,
             params: List[Union[Tuple[Text, Text], Tuple[Text, Text, bool]]]
             ) -> None:
    for p in params:
        none_is_valid = False if len(p) == 2 else p[2]
        validated = check_path_exists(getattr(args, p[0]), p[0], p[1],
                                      none_is_valid)
        setattr(args, p[0], validated)


def parse_last_positional_argument_as_model_path() -> None:
    import sys

    if (len(sys.argv) >= 2 and
            sys.argv[1] in ["run", "test"] and not
            sys.argv[-2].startswith('-') and
            os.path.exists(sys.argv[-1])):
        sys.argv.append(sys.argv[-1])
        sys.argv[-2] = "--model"


def create_default_output_path(model_directory: Text = DEFAULT_MODELS_PATH,
                               prefix: Text = "") -> Text:
    import time

    time_format = "%Y%m%d-%H%M%S"
    return "{}/{}{}.tar.gz".format(model_directory, prefix,
                                time.strftime(time_format))
