from typing import Any, Optional, Text, Union
from pathlib import Path, PurePath

from rasa.shared.exceptions import RasaException
import rasa.shared.utils.io


def relative_normpath(f: Optional[Text], path: Text) -> Optional[Path]:
    """Return the path of file relative to `path`."""
    if f is not None:
        return PurePath(f).relative_to(path)
    return None


def module_path_from_object(o: Any) -> Text:
    """Returns the fully qualified class path of the instantiated object."""
    return o.__class__.__module__ + "." + o.__class__.__name__


def write_json_to_file(filename: Union[Path, Text], obj: Any, **kwargs: Any) -> None:
    """Write an object as a json string to a file."""
    write_to_file(filename, rasa.shared.utils.io.json_to_string(obj, **kwargs))


def write_to_file(filename: Union[Path, Text], text: Any) -> None:
    """Write a text to a file."""

    rasa.shared.utils.io.write_text_file(str(text), filename)


def is_model_dir(model_dir: Text) -> bool:
    """Checks if the given directory contains a model and can be safely removed.

    specifically checks if the directory has no subdirectories and
    if all files have an appropriate ending."""
    allowed_extensions = {".json", ".pkl", ".dat"}

    dir_tree = Path(model_dir)
    if not dir_tree.is_dir():
        return False

    if any([d.is_dir() for d in dir_tree.iterdir()]):  # look for subdirectories
        return False

    file_extensions = [PurePath(f).suffix for f in dir_tree.iterdir()]
    only_valid_files = all([ext in allowed_extensions for ext in file_extensions])
    return only_valid_files


def is_url(resource_name: Text) -> bool:
    """Check whether the url specified is a well formed one.

    Args:
        resource_name: Remote URL to validate

    Returns:
        `True` if valid, otherwise `False`.
    """
    from urllib import parse

    try:
        result = parse.urlparse(resource_name)
    except Exception:
        return False

    if result.scheme == "file":
        return bool(result.path)

    return bool(result.scheme in ["http", "https", "ftp", "ftps"] and result.netloc)


def remove_model(model_dir: Text) -> bool:
    """Removes a model directory and all its content."""
    import shutil

    if is_model_dir(model_dir):
        shutil.rmtree(model_dir)
        return True
    else:
        raise RasaException(
            f"Failed to remove {model_dir}, it seems it is not a model "
            f"directory. E.g. a directory which contains sub directories "
            f"is considered unsafe to remove."
        )
