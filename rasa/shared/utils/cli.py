import math
import shutil
import sys
from typing import Any, Text, NoReturn

import rasa.shared.utils.io


def print_color(*args: Any, color: Text) -> None:
    """Print the given arguments to STDOUT in the specified color.

    Args:
        args: A list of objects to be printed.
        color: A textual representation of the color.
    """
    output = rasa.shared.utils.io.wrap_with_color(*args, color=color)
    stream = sys.stdout
    if sys.platform == "win32":
        # colorama is used to fix a regression where colors can not be printed on
        # windows. https://github.com/RasaHQ/rasa/issues/7053
        from colorama import AnsiToWin32

        stream = AnsiToWin32(sys.stdout).stream
    try:
        print(output, file=stream)
    except BlockingIOError:
        rasa.shared.utils.io.handle_print_blocking(output)


def print_success(*args: Any) -> None:
    """Print the given arguments to STDOUT in green, indicating success.

    Args:
        args: A list of objects to be printed.
    """
    print_color(*args, color=rasa.shared.utils.io.bcolors.OKGREEN)


def print_info(*args: Any) -> None:
    """Print the given arguments to STDOUT in blue.

    Args:
        args: A list of objects to be printed.
    """
    print_color(*args, color=rasa.shared.utils.io.bcolors.OKBLUE)


def print_warning(*args: Any) -> None:
    """Print the given arguments to STDOUT in a color indicating a warning.

    Args:
        args: A list of objects to be printed.
    """
    print_color(*args, color=rasa.shared.utils.io.bcolors.WARNING)


def print_error(*args: Any) -> None:
    """Print the given arguments to STDOUT in a color indicating an error.

    Args:
        args: A list of objects to be printed.
    """
    print_color(*args, color=rasa.shared.utils.io.bcolors.FAIL)


def print_error_and_exit(message: Text, exit_code: int = 1) -> NoReturn:
    """Print an error message and exit the application.

    Args:
        message: The error message to be printed.
        exit_code: The program exit code, defaults to 1.
    """
    print_error(message)
    sys.exit(exit_code)


def pad(text: Text, char: Text = "=", min: int = 3) -> Text:
    """Pad text to a certain length.

    Uses `char` to pad the text to the specified length. If the text is longer
    than the specified length, at least `min` are used.

    The padding is applied to the left and right of the text (almost) equally.

    Example:
        >>> pad("Hello")
        "========= Hello ========"
        >>> pad("Hello", char="-")
        "--------- Hello --------"

    Args:
        text: Text to pad.
        min: Minimum length of the padding.
        char: Character to pad with.

    Returns:
        Padded text.
    """
    width = shutil.get_terminal_size((80, 20)).columns
    padding = max(width - len(text) - 2, min * 2)

    return char * (padding // 2) + " " + text + " " + char * math.ceil(padding / 2)
