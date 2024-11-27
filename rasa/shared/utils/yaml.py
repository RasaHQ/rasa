import datetime
import logging
import os
import re
from collections import OrderedDict
from dataclasses import dataclass
from dataclasses import field
from functools import lru_cache
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Tuple, Union

import jsonschema
from importlib_resources import files
from packaging import version
from pykwalify.core import Core
from pykwalify.errors import SchemaError
from ruamel import yaml as yaml
from ruamel.yaml import RoundTripRepresenter, YAMLError
from ruamel.yaml.constructor import DuplicateKeyError, BaseConstructor, ScalarNode
from ruamel.yaml.comments import CommentedSeq, CommentedMap
from ruamel.yaml.loader import SafeLoader

from rasa.shared.constants import (
    ASSERTIONS_SCHEMA_EXTENSIONS_FILE,
    ASSERTIONS_SCHEMA_FILE,
    MODEL_CONFIG_SCHEMA_FILE,
    CONFIG_SCHEMA_FILE,
    DOCS_URL_TRAINING_DATA,
    PACKAGE_NAME,
    LATEST_TRAINING_DATA_FORMAT_VERSION,
    SCHEMA_EXTENSIONS_FILE,
    RESPONSES_SCHEMA_FILE,
)
from rasa.shared.exceptions import (
    YamlException,
    YamlSyntaxException,
    SchemaValidationError,
    RasaException,
    FileNotFoundException,
)
from rasa.shared.utils.constants import (
    DEFAULT_ENCODING,
    READ_YAML_FILE_CACHE_MAXSIZE_ENV_VAR,
    DEFAULT_READ_YAML_FILE_CACHE_MAXSIZE,
)
from rasa.shared.utils.io import (
    read_file,
    convert_to_ordered_dict,
    raise_warning,
    read_json_file,
)

logger = logging.getLogger(__name__)

KEY_TRAINING_DATA_FORMAT_VERSION = "version"
YAML_VERSION = (1, 2)
READ_YAML_FILE_CACHE_MAXSIZE = os.environ.get(
    READ_YAML_FILE_CACHE_MAXSIZE_ENV_VAR, DEFAULT_READ_YAML_FILE_CACHE_MAXSIZE
)


@dataclass
class PathWithError:
    message: str
    path: List[str] = field(default_factory=list)


def fix_yaml_loader() -> None:
    """Ensure that any string read by yaml is represented as unicode."""

    def construct_yaml_str(self: BaseConstructor, node: ScalarNode) -> Any:
        # Override the default string handling function
        # to always return unicode objects
        return self.construct_scalar(node)

    yaml.Loader.add_constructor("tag:yaml.org,2002:str", construct_yaml_str)
    yaml.SafeLoader.add_constructor("tag:yaml.org,2002:str", construct_yaml_str)


def replace_environment_variables() -> None:
    """Enable yaml loader to process the environment variables in the yaml."""
    # eg. ${USER_NAME}, ${PASSWORD}
    env_var_pattern = re.compile(r"^(.*)\$\{(.*)\}(.*)$")
    yaml.Resolver.add_implicit_resolver("!env_var", env_var_pattern, None)

    def env_var_constructor(loader: BaseConstructor, node: ScalarNode) -> str:
        """Process environment variables found in the YAML."""
        value = loader.construct_scalar(node)
        expanded_vars = os.path.expandvars(value)
        not_expanded = [
            w for w in expanded_vars.split() if w.startswith("$") and w in value
        ]
        if not_expanded:
            raise RasaException(
                f"Error when trying to expand the "
                f"environment variables in '{value}'. "
                f"Please make sure to also set these "
                f"environment variables: '{not_expanded}'."
            )
        return expanded_vars

    yaml.SafeConstructor.add_constructor("!env_var", env_var_constructor)


fix_yaml_loader()
replace_environment_variables()


class YamlValidationException(YamlException, ValueError):
    """Raised if a yaml file does not correspond to the expected schema."""

    def __init__(
        self,
        message: str,
        validation_errors: Optional[List[PathWithError]] = None,
        filename: Optional[str] = None,
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

    def __str__(self) -> str:
        msg = self.file_error_message()
        msg += " Failed to validate YAML. "
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

                error_representation += f"      {error.message}"
                unique_errors[error.message] = error_representation
            error_msg = "\n".join(unique_errors.values())
            msg += f":\n{error_msg}"
        return msg

    def _calculate_number_of_lines(
        self,
        current: Union[CommentedSeq, CommentedMap],
        target: Optional[str] = None,
    ) -> Tuple[int, bool]:
        """Counts the lines that are missing due to the ruamel yaml parser logic.

        Since not all nodes returned from the ruamel yaml parser
        have line numbers attached (arrays have them, dicts have
        them, but strings don't), this method calculates the number
        of lines that are missing instead of just returning line of the parent element

        Args:
        current: current content
        target: target key to find the line number of

        Returns:
            A tuple containing a number of missing lines
            and a flag indicating if an element with a line number was found
        """
        if isinstance(current, list):
            # return the line number of the last list element
            line_number = current[-1].lc.line + 1
            logger.debug(f"Returning from list: last element at {line_number}")
            return line_number, True

        keys_to_check = list(current.keys())
        if target:
            # If target is specified, only check keys before it
            keys_to_check = keys_to_check[: keys_to_check.index(target)]
        try:
            # find the last key that has a line number attached
            last_key_with_lc = next(
                iter(
                    [
                        key
                        for key in reversed(keys_to_check)
                        if hasattr(current[key], "lc")
                    ]
                )
            )
            logger.debug(f"Last key with line number: {last_key_with_lc}")
        except StopIteration:
            # otherwise return the number of elements on that level up to the target
            logger.debug(f"No line number found in {current}")
            if target:
                return list(current.keys()).index(target), False
            return len(list(current.keys())), False

        offset = current[last_key_with_lc].lc.line if not target else 0
        # Recursively calculate the number of lines
        # for the element associated with the last key with a line number
        child_offset, found_lc = self._calculate_number_of_lines(
            current[last_key_with_lc]
        )
        if not found_lc:
            child_offset += offset
        if target:
            child_offset += 1
        # add the number of trailing keys without line numbers to the offset
        last_idx_with_lc = keys_to_check.index(last_key_with_lc)
        child_offset += len(keys_to_check[last_idx_with_lc + 1 :])

        logger.debug(f"Analysed {current}, found {child_offset} lines")
        # Return the calculated child offset and True indicating a line number was found
        return child_offset, True

    def _line_number_for_path(self, current: Any, path: List[str]) -> Optional[int]:
        """Get line number for a yaml path in the current content.

        Implemented using recursion: algorithm goes down the path navigating to the
        leaf in the YAML tree.

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

        head, tail = path[0], path[1:]

        if head == "":
            return current.lc.line

        if head:
            if isinstance(current, dict) and head in current:
                line = self._line_number_for_path(current[head], tail)
                if line is None:
                    line_offset, found_lc = self._calculate_number_of_lines(
                        current, head
                    )
                    if found_lc:
                        return line_offset
                    return this_line + line_offset
                return line
            elif isinstance(current, list) and head.isdigit():
                return self._line_number_for_path(current[int(head)], tail) or this_line
            else:
                return this_line
        return self._line_number_for_path(current, tail) or this_line


def read_schema_file(
    schema_file: str, package_name: str = PACKAGE_NAME
) -> Union[List[Any], Dict[str, Any]]:
    """Read a schema file from the package.

    Args:
        schema_file: The schema file to read.
        package_name: the name of the package the schema is located in. defaults
            to `rasa`.

    Returns:
        The schema as a dictionary.
    """
    schema_path = str(files(package_name).joinpath(schema_file))
    return read_yaml_file(schema_path)


def parse_raw_yaml(raw_yaml_content: str) -> Dict[str, Any]:
    """Parses yaml from a text and raises an exception if the content is not valid.

    Args:
        raw_yaml_content: A raw text containing yaml content.

    Returns:
        The parsed content of the YAML.
        If the content is not valid, a `YamlSyntaxException` will be raised.
    """
    try:
        source_data = read_yaml(raw_yaml_content, reader_type=["safe", "rt"])
    except (YAMLError, DuplicateKeyError) as e:
        raise YamlSyntaxException(underlying_yaml_exception=e)

    return source_data


def validate_yaml_content_using_schema(
    yaml_content: Any,
    schema_content: Union[List[Any], Dict[str, Any]],
    schema_extensions: Optional[List[str]] = None,
) -> None:
    """Validate yaml content using a schema with optional schema extensions.

    Args:
        yaml_content: the content of the YAML to be validated
        schema_content: the content of the YAML schema
        schema_extensions: pykwalify schema extension files
    """
    log = logging.getLogger("pykwalify")
    log.setLevel(logging.CRITICAL)

    core = Core(
        source_data=yaml_content,
        schema_data=schema_content,
        extensions=schema_extensions,
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
            content=yaml_content,
        )


def validate_raw_yaml_using_schema(
    raw_yaml_content: str,
    schema_content: Dict[str, Any],
    schema_extensions: Optional[List[str]] = None,
) -> None:
    """Validate raw yaml content using a schema.

    If the content is not valid, a `YamlSyntaxException` will be raised.

    Args:
        raw_yaml_content: the raw YAML content to be validated (usually a string)
        schema_content: the schema for the yaml_file_content
        schema_extensions: pykwalify schema extension files
    """
    try:
        # we need "rt" since
        # it will add meta information to the parsed output. this meta information
        # will include e.g. at which line an object was parsed. this is very
        # helpful when we validate files later on and want to point the user to the
        # right line
        yaml_data = read_yaml(raw_yaml_content, reader_type=["safe", "rt"])
    except (YAMLError, DuplicateKeyError) as e:
        raise YamlSyntaxException(underlying_yaml_exception=e)

    validate_yaml_content_using_schema(yaml_data, schema_content, schema_extensions)


def validate_raw_yaml_using_schema_file(
    raw_yaml_content: str, schema_path: str, package_name: str = PACKAGE_NAME
) -> None:
    """Validate raw yaml content using a schema from file.

    Args:
        raw_yaml_content: the raw YAML content to be validated (usually a string)
        schema_path: the schema used for validation
        package_name: the name of the package the schema is located in. defaults
            to `rasa`.
    """
    schema_content = read_schema_file(schema_path, package_name)
    validate_raw_yaml_using_schema(raw_yaml_content, schema_content)


def validate_raw_yaml_content_using_schema_with_responses(
    raw_yaml_content: str,
    schema_content: Union[List[Any], Dict[str, Any]],
    package_name: str = PACKAGE_NAME,
) -> None:
    """Validate raw yaml content using a schema with responses sub-schema.

    Args:
        raw_yaml_content: the raw YAML content to be validated (usually a string)
        schema_content: the content of the YAML schema
        package_name: the name of the package the schema is located in. defaults
        to `rasa`.
    """
    # bot responses are part of the schema extension
    # it will be included if the schema explicitly references it with include: responses
    bot_responses_schema_content = read_schema_file(RESPONSES_SCHEMA_FILE, package_name)
    schema_content = dict(schema_content, **bot_responses_schema_content)
    schema_extensions = [str(files(package_name).joinpath(SCHEMA_EXTENSIONS_FILE))]

    validate_raw_yaml_using_schema(raw_yaml_content, schema_content, schema_extensions)


def validate_raw_yaml_using_schema_file_with_responses(
    raw_yaml_content: str, schema_path: str, package_name: str = PACKAGE_NAME
) -> None:
    """Validate domain yaml content using a schema from file with responses sub-schema.

    Args:
        raw_yaml_content: the raw YAML content to be validated (usually a string)
        schema_path: the schema of the yaml file
        package_name: the name of the package the schema is located in. defaults
            to `rasa`.
    """
    schema_content = read_schema_file(schema_path, package_name)
    validate_raw_yaml_content_using_schema_with_responses(
        raw_yaml_content, schema_content, package_name
    )


def read_yaml(
    content: str,
    reader_type: Union[str, List[str]] = "safe",
    **kwargs: Any,
) -> Any:
    """Parses yaml from a text.

    Args:
        content: A text containing yaml content.
        reader_type: Reader type to use. By default, "safe" will be used.
        **kwargs: Any

    Raises:
        ruamel.yaml.parser.ParserError: If there was an error when parsing the YAML.
    """
    custom_constructor = kwargs.get("custom_constructor", None)

    # Create YAML parser with custom constructor
    yaml_parser, reset_constructors = create_yaml_parser(
        reader_type, custom_constructor
    )
    yaml_content = yaml_parser.load(content) or {}

    # Reset to default constructors
    reset_constructors()

    return yaml_content


def create_yaml_parser(
    reader_type: str,
    custom_constructor: Optional[Callable] = None,
) -> Tuple[yaml.YAML, Callable[[], None]]:
    """Create a YAML parser with an optional custom constructor.

    Args:
        reader_type (str): The type of the reader
        (e.g., 'safe', 'rt', 'unsafe').
        custom_constructor (Optional[Callable]):
        A custom constructor function for YAML parsing.

    Returns:
        Tuple[yaml.YAML, Callable[[], None]]: A tuple containing
        the YAML parser and a function to reset constructors to
        their original state.
    """
    yaml_parser = yaml.YAML(typ=reader_type)
    yaml_parser.version = YAML_VERSION  # type: ignore[assignment]
    yaml_parser.preserve_quotes = True  # type: ignore[assignment]

    # Save the original constructors
    original_mapping_constructor = yaml_parser.constructor.yaml_constructors.get(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG
    )
    original_sequence_constructor = yaml_parser.constructor.yaml_constructors.get(
        yaml.resolver.BaseResolver.DEFAULT_SEQUENCE_TAG
    )

    if custom_constructor is not None:
        # Attach the custom constructor to the loader
        yaml_parser.constructor.add_constructor(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, custom_constructor
        )
        yaml_parser.constructor.add_constructor(
            yaml.resolver.BaseResolver.DEFAULT_SEQUENCE_TAG, custom_constructor
        )

    def reset_constructors() -> None:
        """Reset the constructors back to their original state."""
        yaml_parser.constructor.add_constructor(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, original_mapping_constructor
        )
        yaml_parser.constructor.add_constructor(
            yaml.resolver.BaseResolver.DEFAULT_SEQUENCE_TAG,
            original_sequence_constructor,
        )

    def custom_date_constructor(loader: SafeLoader, node: ScalarNode) -> str:
        """Custom constructor for parsing dates in the format '%Y-%m-%d'.

        This constructor parses dates in the '%Y-%m-%d' format and returns them as
        strings instead of datetime objects. This change was introduced because the
        default timestamp constructor in ruamel.yaml returns datetime objects, which
        caused issues in our use case where the `api_version` in the LLM config must
        be a string, but was being interpreted as a datetime object.
        """
        value = loader.construct_scalar(node)
        try:
            # Attempt to parse the date
            date_obj = datetime.datetime.strptime(value, "%Y-%m-%d").date()
            # Return the date as a string instead of a datetime object
            return date_obj.strftime("%Y-%m-%d")
        except ValueError:
            # If the date is not in the correct format, return the original value
            return value

    # Add the custom date constructor
    yaml_parser.constructor.add_constructor(
        "tag:yaml.org,2002:timestamp", custom_date_constructor
    )

    return yaml_parser, reset_constructors


def _is_ascii(text: str) -> bool:
    return all(ord(character) < 128 for character in text)


@lru_cache(maxsize=READ_YAML_FILE_CACHE_MAXSIZE)
def read_yaml_file(
    filename: Union[str, Path], reader_type: Union[str, Tuple[str]] = "safe"
) -> Union[List[Any], Dict[str, Any]]:
    """Parses a yaml file.

    Raises an exception if the content of the file can not be parsed as YAML.

    Args:
        filename: The path to the file which should be read.
        reader_type: Reader type to use. By default "safe" will be used.

    Returns:
        Parsed content of the file.
    """
    try:
        fixed_reader_type = (
            list(reader_type) if isinstance(reader_type, tuple) else reader_type
        )
        return read_yaml(read_file(filename, DEFAULT_ENCODING), fixed_reader_type)
    except (YAMLError, DuplicateKeyError) as e:
        raise YamlSyntaxException(filename, e)


def read_config_file(
    filename: Union[Path, str], reader_type: Union[str, List[str]] = "safe"
) -> Dict[str, Any]:
    """Parses a yaml configuration file. Content needs to be a dictionary.

    Args:
        filename: The path to the file which should be read.
        reader_type: Reader type to use. By default "safe" will be used.

    Raises:
        YamlValidationException: In case file content is not a `Dict`.

    Returns:
        Parsed config file.
    """
    return read_validated_yaml(filename, CONFIG_SCHEMA_FILE, reader_type)


def read_model_configuration(filename: Union[Path, str]) -> Dict[str, Any]:
    """Parses a model configuration file.

    Args:
        filename: The path to the file which should be read.

    Raises:
        YamlValidationException: In case the model configuration doesn't match the
            expected schema.

    Returns:
        Parsed config file.
    """
    return read_validated_yaml(filename, MODEL_CONFIG_SCHEMA_FILE)


def dump_obj_as_yaml_to_string(
    obj: Any, should_preserve_key_order: bool = False
) -> str:
    """Writes data (python dict) to a yaml string.

    Args:
        obj: The object to dump. Has to be serializable.
        should_preserve_key_order: Whether to force preserve key order in `data`.

    Returns:
        The object converted to a YAML string.
    """
    buffer = StringIO()

    write_yaml(obj, buffer, should_preserve_key_order=should_preserve_key_order)

    return buffer.getvalue()


def _enable_ordered_dict_yaml_dumping() -> None:
    """Ensure that `OrderedDict`s are dumped so that the order of keys is respected."""
    yaml.add_representer(
        OrderedDict,
        RoundTripRepresenter.represent_dict,
        representer=RoundTripRepresenter,
    )


YAML_LINE_MAX_WIDTH = 4096


def write_yaml(
    data: Any,
    target: Union[str, Path, StringIO],
    should_preserve_key_order: bool = False,
) -> None:
    """Writes a yaml to the file or to the stream.

    Args:
        data: The data to write.
        target: The path to the file which should be written or a stream object
        should_preserve_key_order: Whether to force preserve key order in `data`.
    """
    _enable_ordered_dict_yaml_dumping()

    if should_preserve_key_order:
        data = convert_to_ordered_dict(data)

    dumper = yaml.YAML()
    # no wrap lines
    dumper.width = YAML_LINE_MAX_WIDTH  # type: ignore[assignment]

    # use `null` to represent `None`
    dumper.representer.add_representer(
        type(None),
        lambda self, _: self.represent_scalar("tag:yaml.org,2002:null", "null"),
    )

    if isinstance(target, StringIO):
        dumper.dump(data, target)
        return

    with Path(target).open("w", encoding=DEFAULT_ENCODING) as outfile:
        dumper.dump(data, outfile)


def is_key_in_yaml(file_path: Union[str, Path], *keys: str) -> bool:
    """Checks if any of the keys is contained in the root object of the yaml file.

    Arguments:
        file_path: path to the yaml file
        keys: keys to look for

    Returns:
          `True` if at least one of the keys is found, `False` otherwise.

    Raises:
        FileNotFoundException: if the file cannot be found.
    """
    try:
        with open(file_path, encoding=DEFAULT_ENCODING) as file:
            return any(
                any(line.lstrip().startswith(f"{key}:") for key in keys)
                for line in file
            )
    except FileNotFoundError:
        raise FileNotFoundException(
            f"Failed to read file, " f"'{os.path.abspath(file_path)}' does not exist."
        )


def read_validated_yaml(
    filename: Union[str, Path],
    schema: str,
    reader_type: Union[str, List[str]] = "safe",
) -> Any:
    """Validates YAML file content and returns parsed content.

    Args:
        filename: The path to the file which should be read.
        schema: The path to the schema file which should be used for validating the
            file content.
        reader_type: Reader type to use. By default, "safe" will be used.

    Returns:
        The parsed file content.

    Raises:
        YamlValidationException: In case the model configuration doesn't match the
            expected schema.
    """
    content = read_file(filename)

    validate_raw_yaml_using_schema_file(content, schema)
    return read_yaml(content, reader_type)


def validate_training_data(json_data: Dict[str, Any], schema: Dict[str, Any]) -> None:
    """Validate rasa training data format to ensure proper training.

    Args:
        json_data: the data to validate
        schema: the schema

    Raises:
        SchemaValidationError if validation fails.
    """
    try:
        jsonschema.validate(json_data, schema)
    except jsonschema.ValidationError as e:
        e.message += (
            f". Failed to validate data, make sure your data "
            f"is valid. For more information about the format visit "
            f"{DOCS_URL_TRAINING_DATA}."
        )
        raise SchemaValidationError.create_from(e) from e


def validate_training_data_format_version(
    yaml_file_content: Dict[str, Any], filename: Optional[str]
) -> bool:
    """Validates version on the training data content using `version` field.

       Warns users if the file is not compatible with the current version of
       Rasa Pro.

    Args:
        yaml_file_content: Raw content of training data file as a dictionary.
        filename: Name of the validated file.

    Returns:
        `True` if the file can be processed by current version of Rasa Pro,
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
            f"Rasa Pro will read the file as a "
            f"version '{LATEST_TRAINING_DATA_FORMAT_VERSION}' file. "
            f"See {DOCS_URL_TRAINING_DATA}."
        )
        return True

    try:
        if isinstance(version_value, str):
            version_value = version_value.strip("\"'")
        parsed_version = version.parse(version_value)
        latest_version = version.parse(LATEST_TRAINING_DATA_FORMAT_VERSION)

        if parsed_version < latest_version:
            raise_warning(
                f"Training data file {filename} has a lower "
                f"format version than your Rasa Pro installation: "
                f"{version_value} < {LATEST_TRAINING_DATA_FORMAT_VERSION}. "
                f"Rasa Pro will read the file as a version "
                f"{LATEST_TRAINING_DATA_FORMAT_VERSION} file. "
                f"Please update your version key to "
                f"{LATEST_TRAINING_DATA_FORMAT_VERSION}. "
                f"See {DOCS_URL_TRAINING_DATA}."
            )

        if latest_version >= parsed_version:
            return True

    except (TypeError, version.InvalidVersion):
        raise_warning(
            f"Training data file {filename} must specify "
            f"'{KEY_TRAINING_DATA_FORMAT_VERSION}' as string, for example:\n"
            f"{KEY_TRAINING_DATA_FORMAT_VERSION}: "
            f"'{LATEST_TRAINING_DATA_FORMAT_VERSION}'\n"
            f"Rasa Pro will read the file as a "
            f"version '{LATEST_TRAINING_DATA_FORMAT_VERSION}' file.",
            docs=DOCS_URL_TRAINING_DATA,
        )
        return True

    raise_warning(
        f"Training data file {filename} has a greater "
        f"format version than your Rasa Pro installation: "
        f"{version_value} > {LATEST_TRAINING_DATA_FORMAT_VERSION}. "
        f"Please consider updating to the latest version of Rasa Pro."
        f"This file will be skipped.",
        docs=DOCS_URL_TRAINING_DATA,
    )
    return False


def default_error_humanizer(error: jsonschema.ValidationError) -> str:
    """Creates a user readable error message for an error."""
    return error.message


def validate_yaml_with_jsonschema(
    yaml_file_content: str,
    schema_path: str,
    package_name: str = PACKAGE_NAME,
    humanize_error: Callable[
        [jsonschema.ValidationError], str
    ] = default_error_humanizer,
) -> None:
    """Validate data format.

    Args:
        yaml_file_content: the content of the yaml file to be validated
        schema_path: the schema of the yaml file
        package_name: the name of the package the schema is located in. defaults
            to `rasa`.
        humanize_error: a function to convert a jsonschema.ValidationError into a
            human-readable error message. Defaults to `default_error_humanizer`.

    Raises:
        YamlSyntaxException: if the yaml file is not valid.
        SchemaValidationError: if validation fails.
    """
    from ruamel.yaml import YAMLError
    import importlib_resources

    schema_file = str(importlib_resources.files(package_name).joinpath(schema_path))
    schema_content = read_json_file(schema_file)

    try:
        # we need "rt" since
        # it will add meta information to the parsed output. this meta information
        # will include e.g. at which line an object was parsed. this is very
        # helpful when we validate files later on and want to point the user to the
        # right line
        source_data = read_yaml(yaml_file_content, reader_type=["safe", "rt"])
    except (YAMLError, DuplicateKeyError) as e:
        raise YamlSyntaxException(underlying_yaml_exception=e)

    try:
        jsonschema.validate(source_data, schema_content)
    except jsonschema.ValidationError as error:
        errors = [
            PathWithError(
                message=humanize_error(error),
                path=[str(e) for e in error.absolute_path],
            )
        ]
        raise YamlValidationException(
            "Please make sure the file is correct and all "
            "mandatory parameters are specified. Here are the errors "
            "found during validation",
            errors,
            content=source_data,
        )


def validate_yaml_data_using_schema_with_assertions(
    yaml_data: Any,
    schema_content: Union[List[Any], Dict[str, Any]],
    package_name: str = PACKAGE_NAME,
) -> None:
    """Validate raw yaml content using a schema with assertions sub-schema.

    Args:
        yaml_data: the parsed yaml data to be validated
        schema_content: the content of the YAML schema
        package_name: the name of the package the schema is located in. defaults
        to `rasa`.
    """
    # test case assertions are part of the schema extension
    # it will be included if the schema explicitly references it with
    # include: assertions
    e2e_test_cases_schema_content = read_schema_file(
        ASSERTIONS_SCHEMA_FILE, package_name
    )

    schema_content = dict(schema_content, **e2e_test_cases_schema_content)
    schema_extensions = [
        str(files(package_name).joinpath(ASSERTIONS_SCHEMA_EXTENSIONS_FILE))
    ]

    validate_yaml_content_using_schema(yaml_data, schema_content, schema_extensions)
