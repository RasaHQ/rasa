import os
from pathlib import Path
from typing import (
    Dict,
    Text,
    Union,
)

import rasa.shared.core.constants
from rasa.shared.exceptions import RasaException, YamlException, YamlSyntaxException
import rasa.shared.nlu.constants
import rasa.shared.utils.validation
import rasa.shared.utils.io
import rasa.shared.utils.common


class InvalidMarkersConfig(RasaException):
    """Exception that can be raised when domain is not valid."""


class MarkerConfig:
    """
    A class that represents the configuration file which lists the different marker
    and the conditions for when each of them applies.
    """

    @classmethod
    def empty_config(cls) -> Dict:
        """Returns an empty config file."""
        return {}  # currently represented as a simple dict. Might change later.

    @classmethod
    def load_config_from_path(cls, path: Union[Text, Path]) -> Dict:
        path = os.path.abspath(path)

        if os.path.isfile(path):
            config = cls.from_file(path)
        elif os.path.isdir(path):
            config = cls.from_directory(path)
        else:
            raise InvalidMarkersConfig(
                "Failed to load markers configuration from '{}'. "
                "File not found!".format(os.path.abspath(path))
            )

        return config

    @classmethod
    def from_file(cls, path: Text) -> Dict:
        """Loads the config from a YAML file."""
        return cls.from_yaml(rasa.shared.utils.io.read_file(path), path)

    @classmethod
    def from_yaml(cls, yaml: Text, original_filename: Text = "") -> Dict:
        """Loads the config from YAML text after validating it."""
        try:
            # TODO impleemnt schema validation
            #  rasa.shared.utils.validation.validate_yaml_schema(yaml,
            #  MAKERS_SCHEMA_FILE)

            data = rasa.shared.utils.io.read_yaml(yaml)
            return data
        except YamlException as e:
            e.filename = original_filename
            raise e

    @classmethod
    def from_directory(cls, path: Text) -> Dict:
        """Loads and appends multiple configs files from a directory tree."""

        config = cls.empty_config()
        for root, _, files in os.walk(path, followlinks=True):
            for file in files:
                full_path = os.path.join(root, file)
                if cls.is_markers_config_file(full_path):
                    other = cls.from_file(full_path)
                    config = cls._merge(config, other)
        return config

    @classmethod
    def _merge(cls, config_a: Dict, config_b: Dict) -> Dict:
        """Merges multiple marker dictionaries"""
        copy_config_a = config_a.copy()
        if "markers" in copy_config_a.keys():
            copy_config_a["markers"].extend(config_b["markers"])
            return copy_config_a
        else:
            return config_b

    @staticmethod
    def is_markers_config_file(filename: Text) -> bool:
        """Checks whether the given file path is a config file.

        Args:
            filename: Path of the file which should be checked.

        Returns:
            `True` if it's a config file, otherwise `False`.

        Raises:
            YamlException: if the file seems to be a YAML file (extension) but
                can not be read / parsed.
        """
        from rasa.shared.data import is_likely_yaml_file

        if not is_likely_yaml_file(filename):
            return False

        try:
            # check if the file can be read as a yaml file.
            rasa.shared.utils.io.read_yaml_file(filename)
        except (RasaException, YamlSyntaxException):
            rasa.shared.utils.io.raise_warning(
                message=f"The file {filename} could not be loaded "
                f"as a markers config file. "
                f"You can use https://yamlchecker.com/ to validate "
                f"the YAML syntax of your file.",
                category=UserWarning,
            )
            return False
        return True
