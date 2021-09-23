import logging
import os
from typing import Text, Dict, List, Optional, Any

from packaging import version
from packaging.version import LegacyVersion
from pykwalify.errors import SchemaError

from ruamel.yaml.constructor import DuplicateKeyError

import rasa.shared
from rasa.shared.exceptions import (
    YamlException,
    YamlSyntaxException,
    SchemaValidationError,
)
import rasa.shared.utils.io
from rasa.shared.constants import (
    DOCS_URL_TRAINING_DATA,
    PACKAGE_NAME,
    LATEST_TRAINING_DATA_FORMAT_VERSION,
    SCHEMA_EXTENSIONS_FILE,
    RESPONSES_SCHEMA_FILE,
)

logger = logging.getLogger(__name__)

KEY_TRAINING_DATA_FORMAT_VERSION = "version"


class YamlValidationException(YamlException, ValueError):
    """Raised if a yaml file does not correspond to the expected schema."""

    def __init__(
        self,
        message: Text,
        validation_errors: Optional[List[SchemaError.SchemaErrorEntry]] = None,
        filename: Optional[Text] = None,
        content: Any = None,
    ) -> None:
        """Create The Error.

        Args:
            message: error message
            validation_errors: validation errors
            filename: name of the file which was validated
            content: yaml content loaded from the file (used for line information)
        """
        super(YamlValidationException, self).__init__(filename)

        self.message = message
        self.validation_errors = validation_errors
        self.content = content

    def __str__(self) -> Text:
        msg = ""
        if self.filename:
            msg += f"Failed to validate '{self.filename}'. "
        else:
            msg += "Failed to validate YAML. "
        msg += self.message
        if self.validation_errors:
            unique_errors = {}
            for error in self.validation_errors:
                line_number = self._line_number_for_path(self.content, error.path)

                if line_number and self.filename:
                    error_representation = f"  in {self.filename}:{line_number}:\n"
                elif line_number:
                    error_representation = f"  in Line {line_number}:\n"
                else:
                    error_representation = ""

                error_representation += f"      {error}"
                unique_errors[str(error)] = error_representation
            error_msg = "\n".join(unique_errors.values())
            msg += f":\n{error_msg}"
        return msg

    def _line_number_for_path(self, current: Any, path: Text) -> Optional[int]:
        """Get line number for a yaml path in the current content.

        Implemented using recursion: algorithm goes down the path navigating to the
        leaf in the YAML tree. Unfortunately, not all nodes returned from the
        ruamel yaml parser have line numbers attached (arrays have them, dicts have
        them), e.g. strings don't have attached line numbers.
        If we arrive at a node that has no line number attached, we'll return the
        line number of the parent - that is as close as it gets.

        Args:
            current: current content
            path: path to traverse within the content

        Returns:
            the line number of the path in the content.
        """
        if not current:
            return None

        this_line = current.lc.line + 1 if hasattr(current, "lc") else None

        if not path:
            return this_line

        if "/" in path:
            head, tail = path.split("/", 1)
        else:
            head, tail = path, ""

        if head:
            if isinstance(current, dict) and head in current:
                return self._line_number_for_path(current[head], tail) or this_line
            elif isinstance(current, list) and head.isdigit():
                return self._line_number_for_path(current[int(head)], tail) or this_line
            else:
                return this_line
        return self._line_number_for_path(current, tail) or this_line


def validate_yaml_schema(yaml_file_content: Text, schema_path: Text) -> None:
    """
    Validate yaml content.

    Args:
        yaml_file_content: the content of the yaml file to be validated
        schema_path: the schema of the yaml file
    """
    from pykwalify.core import Core
    from pykwalify.errors import SchemaError
    from ruamel.yaml import YAMLError
    import pkg_resources
    import logging

    log = logging.getLogger("pykwalify")
    log.setLevel(logging.CRITICAL)

    try:
        # we need "rt" since
        # it will add meta information to the parsed output. this meta information
        # will include e.g. at which line an object was parsed. this is very
        # helpful when we validate files later on and want to point the user to the
        # right line
        source_data = rasa.shared.utils.io.read_yaml(
            yaml_file_content, reader_type=["safe", "rt"]
        )
    except (YAMLError, DuplicateKeyError) as e:
        raise YamlSyntaxException(underlying_yaml_exception=e)

    schema_file = pkg_resources.resource_filename(PACKAGE_NAME, schema_path)
    schema_utils_file = pkg_resources.resource_filename(
        PACKAGE_NAME, RESPONSES_SCHEMA_FILE
    )
    schema_extensions = pkg_resources.resource_filename(
        PACKAGE_NAME, SCHEMA_EXTENSIONS_FILE
    )

    # Load schema content using our YAML loader as `pykwalify` uses a global instance
    # which can fail when used concurrently
    schema_content = rasa.shared.utils.io.read_yaml_file(schema_file)
    schema_utils_content = rasa.shared.utils.io.read_yaml_file(schema_utils_file)
    schema_content = dict(schema_content, **schema_utils_content)

    c = Core(
        source_data=source_data,
        schema_data=schema_content,
        extensions=[schema_extensions],
    )

    try:
        c.validate(raise_exception=True)
    except SchemaError:
        raise YamlValidationException(
            "Please make sure the file is correct and all "
            "mandatory parameters are specified. Here are the errors "
            "found during validation",
            c.errors,
            content=source_data,
        )


def validate_training_data(json_data: Dict[Text, Any], schema: Dict[Text, Any]) -> None:
    """Validate rasa training data format to ensure proper training.

    Args:
        json_data: the data to validate
        schema: the schema

    Raises:
        SchemaValidationError if validation fails.
    """
    from jsonschema import validate
    from jsonschema import ValidationError

    try:
        validate(json_data, schema)
    except ValidationError as e:
        e.message += (
            f". Failed to validate data, make sure your data "
            f"is valid. For more information about the format visit "
            f"{DOCS_URL_TRAINING_DATA}."
        )
        raise SchemaValidationError.create_from(e) from e


def validate_training_data_format_version(
    yaml_file_content: Dict[Text, Any], filename: Optional[Text]
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

    if filename:
        filename = os.path.abspath(filename)

    if not isinstance(yaml_file_content, dict):
        raise YamlValidationException(
            "YAML content in is not a mapping, can not validate training "
            "data schema version.",
            filename=filename,
        )

    version_value = yaml_file_content.get(KEY_TRAINING_DATA_FORMAT_VERSION)

    if not version_value:
        # not raising here since it's not critical
        logger.info(
            f"The '{KEY_TRAINING_DATA_FORMAT_VERSION}' key is missing in "
            f"the training data file {filename}. "
            f"Rasa Open Source will read the file as a "
            f"version '{LATEST_TRAINING_DATA_FORMAT_VERSION}' file. "
            f"See {DOCS_URL_TRAINING_DATA}."
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
            f"{KEY_TRAINING_DATA_FORMAT_VERSION}: "
            f"'{LATEST_TRAINING_DATA_FORMAT_VERSION}'\n"
            f"Rasa Open Source will read the file as a "
            f"version '{LATEST_TRAINING_DATA_FORMAT_VERSION}' file.",
            docs=DOCS_URL_TRAINING_DATA,
        )
        return True

    rasa.shared.utils.io.raise_warning(
        f"Training data file {filename} has a greater "
        f"format version than your Rasa Open Source installation: "
        f"{version_value} > {LATEST_TRAINING_DATA_FORMAT_VERSION}. "
        f"Please consider updating to the latest version of Rasa Open Source."
        f"This file will be skipped.",
        docs=DOCS_URL_TRAINING_DATA,
    )
    return False
