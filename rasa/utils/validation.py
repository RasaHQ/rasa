import logging
from typing import Any, Dict, List, Text, Union

import importlib_resources
import rasa.shared.utils.io
from pykwalify.core import Core
from pykwalify.errors import SchemaError
from rasa.shared.exceptions import YamlSyntaxException
from rasa.shared.utils.validation import PathWithError, YamlValidationException
from ruamel.yaml import YAMLError
from ruamel.yaml.constructor import DuplicateKeyError

from rasa.constants import PACKAGE_NAME


def read_schema_file(schema_file: Text) -> Union[List[Any], Dict[Text, Any]]:
    """Read a schema file from the package.

    Args:
        schema_file: The schema file to read.

    Returns:
        The schema as a dictionary.
    """
    schema_path = str(importlib_resources.files(PACKAGE_NAME).joinpath(schema_file))
    return rasa.shared.utils.io.read_yaml_file(schema_path)


def validate_all_yaml_inputs(
    yaml_inputs: List[Dict[Text, Any]],
    schema_content: Union[List[Any], Dict[Text, Any]],
) -> None:
    """Validate yaml content.

    Args:
        yaml_inputs: list of parsed yaml file contents to be validated
        schema_content: the schema which is used to validate the yaml file
    """
    for yaml_input in yaml_inputs:
        validate_yaml_content(
            yaml_file_content=yaml_input, schema_content=schema_content
        )


def read_yaml(raw_yaml_file_content: Text) -> Dict[Text, Any]:
    """Read a yaml file and returns its parsed content .

    Args:
        raw_yaml_file_content: the raw yaml file content

    Returns:
        The parsed yaml file content.
    """
    try:
        # safe loader loads document without resolving unknown tags
        # we need "rt" since
        # it will add meta information to the parsed output. this meta information
        # will include e.g. at which line an object was parsed. this is very
        # helpful when we validate files later on and want to point the user to the
        # right line
        source_data = rasa.shared.utils.io.read_yaml(
            raw_yaml_file_content, reader_type=["safe", "rt"]
        )
    except (YAMLError, DuplicateKeyError) as e:
        raise YamlSyntaxException(underlying_yaml_exception=e)

    return source_data


def validate_yaml_content(
    yaml_file_content: Dict[Text, Any],
    schema_content: Union[List[Any], Dict[Text, Any]],
) -> None:
    """Validate yaml content.

    Args:
        yaml_file_content: parsed content of the yaml file to be validated
        schema_content: the schema which is used to validate the yaml file
    """
    log = logging.getLogger("pykwalify")
    log.setLevel(logging.CRITICAL)

    core = Core(
        source_data=yaml_file_content,
        schema_data=schema_content,
    )

    try:
        core.validate(raise_exception=True)
    except SchemaError:
        raise YamlValidationException(
            "Please make sure the file is correct and all "
            "mandatory parameters are specified. Here are the errors "
            "found during validation",
            [
                PathWithError(message=str(e), path=e.path.split("/"))
                for e in core.errors
            ],
            content=yaml_file_content,
        )
