from typing import Text, Dict, Any

from ruamel.yaml.constructor import DuplicateKeyError

from rasa.constants import PACKAGE_NAME, DOCS_URL_TRAINING_DATA_NLU


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
    import rasa.utils.io
    import logging

    log = logging.getLogger("pykwalify")
    if show_validation_errors:
        log.setLevel(logging.WARN)
    else:
        log.setLevel(logging.CRITICAL)

    try:
        source_data = rasa.utils.io.read_yaml(yaml_file_content)
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

        c = Core(source_data=source_data, schema_files=[schema_file])
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
