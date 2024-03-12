import os
from pathlib import Path
from typing import Any, Dict, List, Text, Union

from rasa.shared.data import is_likely_yaml_file
from rasa.shared.exceptions import RasaException, YamlSyntaxException
from rasa.shared.utils.yaml import read_yaml_file


def collect_yaml_files_from_path(path: Union[Text, Path]) -> List[Text]:
    path = os.path.abspath(path)
    if os.path.isfile(path):
        yaml_files = [
            yaml_file for yaml_file in [path] if is_likely_yaml_file(yaml_file)
        ]
        if not yaml_files:
            raise FileNotFoundError(f"Could not find a yaml file at '{path}'.")
    elif os.path.isdir(path):
        yaml_files = [
            os.path.join(root, file)
            for root, _, files in os.walk(path, followlinks=True)
            for file in files
            if is_likely_yaml_file(file)
        ]
        if not yaml_files:
            raise FileNotFoundError(
                f"Could not find any yaml in the directory tree rooted at '{path}'."
            )
    else:
        raise RasaException(
            f"The given path ({path}) is neither pointing to a directory "
            f"nor a file. Please specify the location of a yaml file or a "
            f"root directory (all yaml configs found in the directories "
            f"under that root directory will be loaded). "
        )
    return yaml_files


YAML_CONFIG = Dict[Text, Any]
YAML_CONFIGS = Dict[Text, Dict[Text, YAML_CONFIG]]


def collect_configs_from_yaml_files(yaml_files: List[Text]) -> YAML_CONFIGS:
    loaded_configs: YAML_CONFIGS = {}
    for yaml_file in yaml_files:
        loaded_config = read_yaml_file(yaml_file)
        if not isinstance(loaded_config, dict):
            raise YamlSyntaxException(
                f"Expected the loaded configurations to be a "
                f"valid YAML dictionary but found a "
                f"{type(loaded_config)} in {yaml_file}. "
            )
        loaded_configs[yaml_file] = loaded_config
    return loaded_configs
