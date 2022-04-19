import asyncio
import filecmp
import logging
import os
import pickle
import tempfile
import warnings
import re
from asyncio import AbstractEventLoop
from pathlib import Path
from typing import Text, Any, Union, List, Type, Callable, TYPE_CHECKING, Pattern
from typing_extensions import Protocol

import rasa.shared.constants
import rasa.shared.utils.io

if TYPE_CHECKING:
    from prompt_toolkit.validation import Validator


class WriteRow(Protocol):
    """Describes a csv writer supporting a `writerow` method (workaround for typing)."""

    def writerow(self, row: List[Text]) -> None:
        """Write the given row.

        Args:
            row: the entries of a row as a list of strings
        """
        ...


def configure_colored_logging(loglevel: Text) -> None:
    """Configures coloredlogs library for specified loglevel.

    Args:
        loglevel: The loglevel to configure the library for
    """
    import coloredlogs

    loglevel = loglevel or os.environ.get(
        rasa.shared.constants.ENV_LOG_LEVEL, rasa.shared.constants.DEFAULT_LOG_LEVEL
    )

    field_styles = coloredlogs.DEFAULT_FIELD_STYLES.copy()
    field_styles["asctime"] = {}
    level_styles = coloredlogs.DEFAULT_LEVEL_STYLES.copy()
    level_styles["debug"] = {}
    coloredlogs.install(
        level=loglevel,
        use_chroot=False,
        fmt="%(asctime)s %(levelname)-8s %(name)s  - %(message)s",
        level_styles=level_styles,
        field_styles=field_styles,
    )


def enable_async_loop_debugging(
    event_loop: AbstractEventLoop, slow_callback_duration: float = 0.1
) -> AbstractEventLoop:
    """Enables debugging on an event loop.

    Args:
        event_loop: The event loop to enable debugging on
        slow_callback_duration: The threshold at which a callback should be
                                alerted as slow.
    """
    logging.info(
        "Enabling coroutine debugging. Loop id {}.".format(id(asyncio.get_event_loop()))
    )

    # Enable debugging
    event_loop.set_debug(True)

    # Make the threshold for "slow" tasks very very small for
    # illustration. The default is 0.1 (= 100 milliseconds).
    event_loop.slow_callback_duration = slow_callback_duration

    # Report all mistakes managing asynchronous resources.
    warnings.simplefilter("always", ResourceWarning)
    return event_loop


def pickle_dump(filename: Union[Text, Path], obj: Any) -> None:
    """Saves object to file.

    Args:
        filename: the filename to save the object to
        obj: the object to store
    """
    with open(filename, "wb") as f:
        pickle.dump(obj, f)


def pickle_load(filename: Union[Text, Path]) -> Any:
    """Loads an object from a file.

    Args:
        filename: the filename to load the object from

    Returns: the loaded object
    """
    with open(filename, "rb") as f:
        return pickle.load(f)


def create_temporary_file(data: Any, suffix: Text = "", mode: Text = "w+") -> Text:
    """Creates a tempfile.NamedTemporaryFile object for data."""
    encoding = None if "b" in mode else rasa.shared.utils.io.DEFAULT_ENCODING
    f = tempfile.NamedTemporaryFile(
        mode=mode, suffix=suffix, delete=False, encoding=encoding
    )
    f.write(data)

    f.close()
    return f.name


def create_temporary_directory() -> Text:
    """Creates a tempfile.TemporaryDirectory."""
    f = tempfile.TemporaryDirectory()
    return f.name


def create_path(file_path: Text) -> None:
    """Makes sure all directories in the 'file_path' exists."""

    parent_dir = os.path.dirname(os.path.abspath(file_path))
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)


def file_type_validator(
    valid_file_types: List[Text], error_message: Text
) -> Type["Validator"]:
    """Creates a `Validator` class which can be used with `questionary` to validate
    file paths.
    """

    def is_valid(path: Text) -> bool:
        return path is not None and any(
            [path.endswith(file_type) for file_type in valid_file_types]
        )

    return create_validator(is_valid, error_message)


def not_empty_validator(error_message: Text) -> Type["Validator"]:
    """Creates a `Validator` class which can be used with `questionary` to validate
    that the user entered something other than whitespace.
    """

    def is_valid(input: Text) -> bool:
        return input is not None and input.strip() != ""

    return create_validator(is_valid, error_message)


def create_validator(
    function: Callable[[Text], bool], error_message: Text
) -> Type["Validator"]:
    """Helper method to create `Validator` classes from callable functions. Should be
    removed when questionary supports `Validator` objects."""

    from prompt_toolkit.validation import Validator, ValidationError
    from prompt_toolkit.document import Document

    class FunctionValidator(Validator):
        @staticmethod
        def validate(document: Document) -> None:
            is_valid = function(document.text)
            if not is_valid:
                raise ValidationError(message=error_message)

    return FunctionValidator


def json_unpickle(
    file_name: Union[Text, Path], encode_non_string_keys: bool = False
) -> Any:
    """Unpickle an object from file using json.

    Args:
        file_name: the file to load the object from
        encode_non_string_keys: If set to `True` then jsonpickle will encode non-string
          dictionary keys instead of coercing them into strings via `repr()`.

    Returns: the object
    """
    import jsonpickle.ext.numpy as jsonpickle_numpy
    import jsonpickle

    jsonpickle_numpy.register_handlers()

    file_content = rasa.shared.utils.io.read_file(file_name)
    return jsonpickle.loads(file_content, keys=encode_non_string_keys)


def json_pickle(
    file_name: Union[Text, Path], obj: Any, encode_non_string_keys: bool = False
) -> None:
    """Pickle an object to a file using json.

    Args:
        file_name: the file to store the object to
        obj: the object to store
        encode_non_string_keys: If set to `True` then jsonpickle will encode non-string
          dictionary keys instead of coercing them into strings via `repr()`.
    """
    import jsonpickle.ext.numpy as jsonpickle_numpy
    import jsonpickle

    jsonpickle_numpy.register_handlers()

    rasa.shared.utils.io.write_text_file(
        jsonpickle.dumps(obj, keys=encode_non_string_keys), file_name
    )


def get_emoji_regex() -> Pattern:
    """Returns regex to identify emojis."""
    return re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\u200d"  # zero width joiner
        "\u200c"  # zero width non-joiner
        "]+",
        flags=re.UNICODE,
    )


def are_directories_equal(dir1: Path, dir2: Path) -> bool:
    """Compares two directories recursively.

    Files in each directory are
    assumed to be equal if their names and contents are equal.

    Args:
        dir1: The first directory.
        dir2: The second directory.

    Returns:
        `True` if they are equal, `False` otherwise.
    """
    dirs_cmp = filecmp.dircmp(dir1, dir2)
    if dirs_cmp.left_only or dirs_cmp.right_only:
        return False

    (_, mismatches, errors) = filecmp.cmpfiles(
        dir1, dir2, dirs_cmp.common_files, shallow=False
    )

    if mismatches or errors:
        return False

    for common_dir in dirs_cmp.common_dirs:
        new_dir1 = Path(dir1, common_dir)
        new_dir2 = Path(dir2, common_dir)

        is_equal = are_directories_equal(new_dir1, new_dir2)
        if not is_equal:
            return False

    return True
