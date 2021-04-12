from typing import Any, Text

import rasa.shared.utils.io


def module_path_from_object(o: Any) -> Text:
    """Returns the fully qualified class path of the instantiated object."""
    return o.__class__.__module__ + "." + o.__class__.__name__


def write_json_to_file(filename: Text, obj: Any, **kwargs: Any) -> None:
    """Write an object as a json string to a file."""

    write_to_file(filename, rasa.shared.utils.io.json_to_string(obj, **kwargs))


def write_to_file(filename: Text, text: Any) -> None:
    """Write a text to a file."""

    rasa.shared.utils.io.write_text_file(str(text), filename)


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
