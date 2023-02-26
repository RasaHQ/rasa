import fcntl
import os
import sys
from typing import Any, Text, NoReturn

import rasa.shared.utils.io


def print_blocking(string) -> None:
    """Save fcntl settings and restore them after print()."""
    save = fcntl.fcntl(sys.stdout.fileno(), fcntl.F_GETFL)
    new = save & ~os.O_NONBLOCK
    fcntl.fcntl(sys.stdout.fileno(), fcntl.F_SETFL, new)
    print(string)
    fcntl.fcntl(sys.stdout.fileno(), fcntl.F_SETFL, save)
    sys.stdout.flush()


def print_color(*args: Any, color: Text) -> None:
    """Print the given arguments in the specified color.

    Args:
        *args: Any type of arguments to be printed.
        color: Text representation of the color.
    """
    output = rasa.shared.utils.io.wrap_with_color(*args, color=color)
    try:
        # colorama is used to fix a regression where colors can not be printed on
        # windows. https://github.com/RasaHQ/rasa/issues/7053
        from colorama import AnsiToWin32

        stream = AnsiToWin32(sys.stdout).stream
        print(output, file=stream)
    except ImportError:
        try:
            print(output)
        except BlockingIOError:
            print_blocking(output)


def print_success(*args: Any) -> None:
    print_color(*args, color=rasa.shared.utils.io.bcolors.OKGREEN)


def print_info(*args: Any) -> None:
    print_color(*args, color=rasa.shared.utils.io.bcolors.OKBLUE)


def print_warning(*args: Any) -> None:
    print_color(*args, color=rasa.shared.utils.io.bcolors.WARNING)


def print_error(*args: Any) -> None:
    print_color(*args, color=rasa.shared.utils.io.bcolors.FAIL)


def print_error_and_exit(message: Text, exit_code: int = 1) -> NoReturn:
    """Print error message and exit the application."""

    print_error(message)
    sys.exit(exit_code)
