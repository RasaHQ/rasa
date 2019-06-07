from typing import Text

from ruamel.yaml.constructor import DuplicateKeyError


class InvalidYamlFileError(ValueError):
    """Raised if an invalid yaml file was provided."""

    def __init__(self, message: Text) -> None:
        super(InvalidYamlFileError, self).__init__(message)


def validate_pipeline_yaml(yaml_file_content: Text, schema_path: Text) -> None:
    from pykwalify.core import Core
    from pykwalify.errors import SchemaError
    from ruamel.yaml import YAMLError
    import pkg_resources
    import rasa.utils.io
    import logging

    log = logging.getLogger("pykwalify")
    log.setLevel(logging.WARN)

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
            "The provided yaml file contains a duplicated key: {}. You can use "
            "http://www.yamllint.com/ to validate the yaml syntax "
            "of your file.".format(str(e))
        )

    try:
        schema_file = pkg_resources.resource_filename("rasa", schema_path)

        c = Core(source_data=source_data, schema_files=[schema_file])
        c.validate(raise_exception=True)
    except SchemaError:
        raise InvalidYamlFileError(
            "Failed to validate yaml file. "
            "Please make sure the file is correct; to do so, "
            "take a look at the errors logged during "
            "validation previous to this exception. "
            "You can also validate the yaml "
            "syntax of our file using http://www.yamllint.com/."
        )
