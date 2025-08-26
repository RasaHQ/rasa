import asyncio
import filecmp
import logging
import os
import re
import tempfile
import warnings
from asyncio import AbstractEventLoop
from collections import OrderedDict
from io import StringIO
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    List,
    Optional,
    Text,
    Type,
    Union,
)

from ruamel import yaml
from ruamel.yaml import RoundTripRepresenter
from typing_extensions import Protocol

import rasa.shared.constants
import rasa.shared.utils.io
from rasa.shared.exceptions import RasaException

if TYPE_CHECKING:
    from prompt_toolkit.validation import Validator

YAML_LINE_MAX_WIDTH = 4096


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
    """Creates a file type validator class for the questionary package."""

    def is_valid(path: Text) -> bool:
        return path is not None and any(
            [path.endswith(file_type) for file_type in valid_file_types]
        )

    return create_validator(is_valid, error_message)


def not_empty_validator(error_message: Text) -> Type["Validator"]:
    """Creates a not empty validator class for the questionary package."""

    def is_valid(input: Text) -> bool:
        return input is not None and input.strip() != ""

    return create_validator(is_valid, error_message)


def create_validator(
    function: Callable[[Text], bool], error_message: Text
) -> Type["Validator"]:
    """Helper method to create a validator class from a callable function."""
    from prompt_toolkit.document import Document
    from prompt_toolkit.validation import ValidationError, Validator

    class FunctionValidator(Validator):
        @staticmethod
        def validate(document: Document) -> None:
            is_valid = function(document.text)
            if not is_valid:
                raise ValidationError(message=error_message)

    return FunctionValidator


def remove_emojis(s: str) -> str:
    import demoji

    replaced = demoji.replace(s)
    if replaced == s:
        return s
    # remove duplicate or trailing whitespaces if emojis were removed
    return re.sub(r" +", " ", replaced).strip()


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


def write_yaml(
    data: Any,
    target: Union[Text, Path, StringIO],
    transform: Optional[Callable] = None,
    should_preserve_key_order: bool = False,
) -> None:
    """Writes a yaml to the file or to the stream.

    Args:
        data: The data to write.
        target: The path to the file which should be written or a stream object
        transform: A function to transform the data before writing it.
        should_preserve_key_order: Whether to force preserve key order in `data`.
    """
    yaml.add_representer(
        OrderedDict,
        RoundTripRepresenter.represent_dict,
        representer=RoundTripRepresenter,
    )

    if should_preserve_key_order:
        data = rasa.shared.utils.io.convert_to_ordered_dict(data)

    dumper = yaml.YAML()
    # no wrap lines
    dumper.width = YAML_LINE_MAX_WIDTH  # type: ignore

    # use `null` to represent `None`
    dumper.representer.add_representer(
        type(None),
        lambda self, _: self.represent_scalar("tag:yaml.org,2002:null", "null"),
    )

    if isinstance(target, StringIO):
        dumper.dump(data, target, transform=transform)
        return

    with Path(target).open("w", encoding="utf-8") as outfile:
        dumper.dump(data, outfile, transform=transform)


class InvalidPathException(RasaException):
    """Raised if a path is invalid - e.g. path traversal is detected."""


def subpath(parent: str, child: str) -> str:
    """Return the path to the child directory of the parent directory.

    Ensures, that child doesn't navigate to parent directories. Prevents
    path traversal. Raises an InvalidPathException if the path is invalid.

    Based on Snyk's directory traversal mitigation:
    https://learn.snyk.io/lesson/directory-traversal/
    """
    safe_path = os.path.abspath(os.path.join(parent, child))
    parent = os.path.abspath(parent)

    common_base = os.path.commonpath([parent, safe_path])
    if common_base != parent:
        raise InvalidPathException(f"Invalid path: {safe_path}")

    return safe_path
