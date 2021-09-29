import os
from pathlib import Path
from typing import (
    Dict,
    Text,
    Union,
)
from ruamel.yaml.parser import ParserError
import rasa.shared.core.constants
from rasa.shared.exceptions import RasaException
import rasa.shared.nlu.constants
import rasa.shared.utils.validation
import rasa.shared.utils.io
import rasa.shared.utils.common
from rasa.shared.data import is_likely_yaml_file
from rasa.shared.utils.schemas.markers import MARKERS_SCHEMA


class InvalidMarkersConfig(RasaException):
    """Exception that can be raised when markers config is not valid."""


class MarkerConfig:
    """A class that represents the markers config.

    A markers config contains the markers and the conditions for when they apply.
    The class reads the config, validates the schema, and validates the conditions.
    """

    @classmethod
    def empty_config(cls) -> Dict:
        """Returns an empty config file."""
        return {}

    @classmethod
    def load_config_from_path(cls, path: Union[Text, Path]) -> Dict:
        """Loads the config from a file or directory."""
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
    def from_yaml(cls, yaml: Text, filename: Text = "") -> Dict:
        """Loads the config from YAML text after validating it."""
        try:

            config = rasa.shared.utils.io.read_yaml(yaml)
            cls._validate_config(config)
            return config

        except ParserError as e:
            e.filename = filename
            rasa.shared.utils.io.raise_warning(
                message=f"The file {filename} could not be loaded "
                f"as a markers config file. "
                f"You can use https://yamlchecker.com/ to validate "
                f"the YAML syntax of the file.",
                category=UserWarning,
            )
            raise e

    @classmethod
    def from_directory(cls, path: Text) -> Dict:
        """Loads and appends multiple configs from a directory tree."""
        combined_configs = cls.empty_config()
        for root, _, files in os.walk(path, followlinks=True):
            for file in files:
                full_path = os.path.join(root, file)
                if is_likely_yaml_file(full_path):
                    config = cls.from_file(full_path)
                    combined_configs = cls._merge(combined_configs, config)
        return combined_configs

    @classmethod
    def _merge(cls, config_a: Dict, config_b: Dict) -> Dict:
        """Merges multiple marker configs."""
        copy_config_a = config_a.copy()
        if "markers" in copy_config_a.keys():
            copy_config_a["markers"].extend(config_b["markers"])
            return copy_config_a
        else:
            return config_b

    @classmethod
    def _validate_config(cls, config: Dict, filename: Text = "") -> bool:
        from jsonschema import validate
        from jsonschema import ValidationError

        try:
            validate(config, MARKERS_SCHEMA)
            return True
        except ValidationError as e:
            e.message += f". The file {filename} is invalid according to the schema."
            raise e

    @staticmethod
    def config_format_spec() -> Dict:
        """Expected schema for a markers config."""
        return MARKERS_SCHEMA
