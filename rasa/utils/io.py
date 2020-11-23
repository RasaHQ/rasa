import asyncio
import logging
import os
import pickle
import tarfile
import tempfile
import warnings
import zipfile
from asyncio import AbstractEventLoop
from io import BytesIO as IOReader
from pathlib import Path
from typing import Text, Any, Union, List, Type, Callable, TYPE_CHECKING

import rasa.shared.constants
import rasa.shared.utils.io

if TYPE_CHECKING:
    from prompt_toolkit.validation import Validator


def configure_colored_logging(loglevel: Text) -> None:
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


def unarchive(byte_array: bytes, directory: Text) -> Text:
    """Tries to unpack a byte array interpreting it as an archive.

    Tries to use tar first to unpack, if that fails, zip will be used."""

    try:
        tar = tarfile.open(fileobj=IOReader(byte_array))
        tar.extractall(directory)
        tar.close()
        return directory
    except tarfile.TarError:
        zip_ref = zipfile.ZipFile(IOReader(byte_array))
        zip_ref.extractall(directory)
        zip_ref.close()
        return directory


def create_temporary_file(data: Any, suffix: Text = "", mode: Text = "w+") -> Text:
    """Creates a tempfile.NamedTemporaryFile object for data.

    mode defines NamedTemporaryFile's  mode parameter in py3."""

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


def zip_folder(folder: Text) -> Text:
    """Create an archive from a folder."""
    import shutil

    zipped_path = tempfile.NamedTemporaryFile(delete=False)
    zipped_path.close()

    # WARN: not thread-safe!
    return shutil.make_archive(zipped_path.name, "zip", folder)


def json_unpickle(file_name: Union[Text, Path]) -> Any:
    """Unpickle an object from file using json.

    Args:
        file_name: the file to load the object from

    Returns: the object
    """
    import jsonpickle.ext.numpy as jsonpickle_numpy
    import jsonpickle

    jsonpickle_numpy.register_handlers()

    file_content = rasa.shared.utils.io.read_file(file_name)
    return jsonpickle.loads(file_content)


def json_pickle(file_name: Union[Text, Path], obj: Any) -> None:
    """Pickle an object to a file using json.

    Args:
        file_name: the file to store the object to
        obj: the object to store
    """
    import jsonpickle.ext.numpy as jsonpickle_numpy
    import jsonpickle

    jsonpickle_numpy.register_handlers()

    rasa.shared.utils.io.write_text_file(jsonpickle.dumps(obj), file_name)
