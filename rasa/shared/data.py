from pathlib import Path
from typing import Text


YAML_FILE_EXTENSIONS = {".yml", ".yaml"}
JSON_FILE_EXTENSIONS = {".json"}
MARKDOWN_FILE_EXTENSIONS = {".md"}
TRAINING_DATA_EXTENSIONS = JSON_FILE_EXTENSIONS.union(MARKDOWN_FILE_EXTENSIONS).union(
    YAML_FILE_EXTENSIONS
)


def is_likely_yaml_file(file_path: Text) -> bool:
    """Check if a file likely contains yaml.

    Arguments:
        file_path: path to the file

    Returns:
        `True` if the file likely contains data in yaml format, `False` otherwise.
    """
    return Path(file_path).suffix in YAML_FILE_EXTENSIONS


def is_likely_json_file(file_path: Text) -> bool:
    """Check if a file likely contains json.

    Arguments:
        file_path: path to the file

    Returns:
        `True` if the file likely contains data in json format, `False` otherwise.
    """
    return Path(file_path).suffix in JSON_FILE_EXTENSIONS


def is_likely_markdown_file(file_path: Text) -> bool:
    """Check if a file likely contains markdown.

    Arguments:
        file_path: path to the file

    Returns:
        `True` if the file likely contains data in markdown format,
        `False` otherwise.
    """
    return Path(file_path).suffix in MARKDOWN_FILE_EXTENSIONS
