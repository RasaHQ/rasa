from pathlib import Path
from typing import Text, Union
import rasa.shared.data
import rasa.shared.utils.io


KEY_FLOWS = "flows"


def is_flows_file(file_path: Union[Text, Path]) -> bool:
    """Check if file contains Flow training data.

    Args:
        file_path: Path of the file to check.

    Returns:
        `True` in case the file is a flows YAML training data file,
        `False` otherwise.

    Raises:
        YamlException: if the file seems to be a YAML file (extension) but
            can not be read / parsed.
    """
    return rasa.shared.data.is_likely_yaml_file(
        file_path
    ) and rasa.shared.utils.io.is_key_in_yaml(file_path, KEY_FLOWS)
