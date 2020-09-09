import logging
from typing import Text, Dict, Any

from packaging import version
from packaging.version import LegacyVersion

from ruamel.yaml.constructor import DuplicateKeyError

import rasa.shared
import rasa.shared.utils.io
from rasa.shared.constants import (
    DOCS_URL_TRAINING_DATA_NLU,
    PACKAGE_NAME,
    LATEST_TRAINING_DATA_FORMAT_VERSION,
    DOCS_BASE_URL,
    SCHEMA_EXTENSIONS_FILE,
    RESPONSES_SCHEMA_FILE,
)

logger = logging.getLogger(__name__)

KEY_TRAINING_DATA_FORMAT_VERSION = "version"


class InvalidYamlFileError(ValueError):
    """Raised if an invalid yaml file was provided."""

    def __init__(self, message: Text) -> None:
        super().__init__(message)


def validate_yaml_schema(
    yaml_file_content: Text, schema_path: Text, show_validation_errors: bool = True
) -> None:
    """
    Validate yaml content.

    Args:
        yaml_file_content: the content of the yaml file to be validated
        schema_path: the schema of the yaml file
        show_validation_errors: if true, validation errors are shown
    """
    from pykwalify.core import Core
    from pykwalify.errors import SchemaError
    from ruamel.yaml import YAMLError
    import pkg_resources
    import logging

    log = logging.getLogger("pykwalify")
    if show_validation_errors:
        log.setLevel(logging.WARN)
    else:
        log.setLevel(logging.CRITICAL)

    try:
        source_data = rasa.shared.utils.io.read_yaml(yaml_file_content)
    except YAMLError:
        raise InvalidYamlFileError(
            "The provided yaml file is invalid. You can use "
            "http://www.yamllint.com/ to validate the yaml syntax "
            "of your file."
        )
    except DuplicateKeyError as e:
        raise InvalidYamlFileError(
            "The provided yaml file contains a duplicated key: '{}'. You can use "
            "http://www.yamllint.com/ to validate the yaml syntax "
            "of your file.".format(str(e))
        )

    try:
        schema_file = pkg_resources.resource_filename(PACKAGE_NAME, schema_path)
        schema_utils_file = pkg_resources.resource_filename(
            PACKAGE_NAME, RESPONSES_SCHEMA_FILE
        )
        schema_extensions = pkg_resources.resource_filename(
            PACKAGE_NAME, SCHEMA_EXTENSIONS_FILE
        )

        c = Core(
            source_data=source_data,
            schema_files=[schema_file, schema_utils_file],
            extensions=[schema_extensions],
        )
        c.validate(raise_exception=True)
    except SchemaError:
        raise InvalidYamlFileError(
            "Failed to validate yaml file. "
            "Please make sure the file is correct and all "
            "mandatory parameters are specified; to do so, "
            "take a look at the errors logged during "
            "validation previous to this exception."
        )


def validate_training_data(json_data: Dict[Text, Any], schema: Dict[Text, Any]) -> None:
    """Validate rasa training data format to ensure proper training.

    Args:
        json_data: the data to validate
        schema: the schema

    Raises:
        ValidationError if validation fails.
    """
    from jsonschema import validate
    from jsonschema import ValidationError

    try:
        validate(json_data, schema)
    except ValidationError as e:
        e.message += (
            f". Failed to validate data, make sure your data "
            f"is valid. For more information about the format visit "
            f"{DOCS_URL_TRAINING_DATA_NLU}."
        )
        raise e


def validate_training_data_format_version(
    yaml_file_content: Dict[Text, Any], filename: Text
) -> bool:
    """Validates version on the training data content using `version` field
       and warns users if the file is not compatible with the current version of
       Rasa Open Source.

    Args:
        yaml_file_content: Raw content of training data file as a dictionary.
        filename: Name of the validated file.

    Returns:
        `True` if the file can be processed by current version of Rasa Open Source,
        `False` otherwise.
    """
    if not isinstance(yaml_file_content, dict):
        raise ValueError(f"Failed to validate {filename}.")

    version_value = yaml_file_content.get(KEY_TRAINING_DATA_FORMAT_VERSION)

    if not version_value:
        # not raising here since it's not critical
        logger.warning(
            f"Training data file {filename} doesn't have a "
            f"'{KEY_TRAINING_DATA_FORMAT_VERSION}' key. "
            f"Rasa Open Source will read the file as a "
            f"version '{LATEST_TRAINING_DATA_FORMAT_VERSION}' file. "
            f"See {DOCS_BASE_URL}."
        )
        return True

    try:
        parsed_version = version.parse(version_value)
        if isinstance(parsed_version, LegacyVersion):
            raise TypeError

        if version.parse(LATEST_TRAINING_DATA_FORMAT_VERSION) >= parsed_version:
            return True

    except TypeError:
        rasa.shared.utils.io.raise_warning(
            f"Training data file {filename} must specify "
            f"'{KEY_TRAINING_DATA_FORMAT_VERSION}' as string, for example:\n"
            f"{KEY_TRAINING_DATA_FORMAT_VERSION}: '{LATEST_TRAINING_DATA_FORMAT_VERSION}'\n"
            f"Rasa Open Source will read the file as a "
            f"version '{LATEST_TRAINING_DATA_FORMAT_VERSION}' file.",
            docs=DOCS_BASE_URL,
        )
        return True

    rasa.shared.utils.io.raise_warning(
        f"Training data file {filename} has a greater format version than "
        f"your Rasa Open Source installation: "
        f"{version_value} > {LATEST_TRAINING_DATA_FORMAT_VERSION}. "
        f"Please consider updating to the latest version of Rasa Open Source."
        f"This file will be skipped.",
        docs=DOCS_BASE_URL,
    )
    return False
