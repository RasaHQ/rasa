import sys
from typing import Any, Text, NoReturn

import rasa.shared.utils.io


def print_color(*args: Any, color: Text) -> None:
    output = rasa.shared.utils.io.wrap_with_color(*args, color=color)
    try:
        # colorama is used to fix a regression where colors can not be printed on
        # windows. https://github.com/RasaHQ/rasa/issues/7053
        from colorama import AnsiToWin32

        stream = AnsiToWin32(sys.stdout).stream
        print(output, file=stream)
    except ImportError:
        print(output)


def print_success(*args: Any):
    print_color(*args, color=rasa.shared.utils.io.bcolors.OKGREEN)


def print_info(*args: Any):
    print_color(*args, color=rasa.shared.utils.io.bcolors.OKBLUE)


def print_warning(*args: Any):
    print_color(*args, color=rasa.shared.utils.io.bcolors.WARNING)


def print_error(*args: Any):
    print_color(*args, color=rasa.shared.utils.io.bcolors.FAIL)


def print_error_and_exit(message: Text, exit_code: int = 1) -> NoReturn:
    """Print error message and exit the application."""

    print_error(message)
    sys.exit(exit_code)
