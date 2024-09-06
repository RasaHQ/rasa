import textwrap
from pathlib import Path
from typing import Any, Dict, List, Text, Union, Optional

import jsonschema
import ruamel.yaml.nodes as yaml_nodes
from ruamel import yaml as yaml

import rasa.shared
import rasa.shared.data
import rasa.shared.utils.io
from rasa.shared.core.flows.flow import Flow
from rasa.shared.core.flows.flows_list import FlowsList
from rasa.shared.exceptions import RasaException, YamlException
from rasa.shared.importers.importer import FlowSyncImporter
from rasa.shared.utils.yaml import (
    validate_yaml_with_jsonschema,
    read_yaml,
    dump_obj_as_yaml_to_string,
    is_key_in_yaml,
)

FLOWS_SCHEMA_FILE = "shared/core/flows/flows_yaml_schema.json"
KEY_FLOWS = "flows"


class YAMLFlowsReader:
    """Class that reads flows information in YAML format."""

    @classmethod
    def read_from_file(
        cls, filename: Union[Text, Path], add_line_numbers: bool = True
    ) -> FlowsList:
        """Read flows from file.

        Args:
            filename: Path to the flows file.
            add_line_numbers: Flag whether to add line numbers to yaml

        Returns:
            `Flow`s read from `filename`.
        """
        try:
            return cls.read_from_string(
                rasa.shared.utils.io.read_file(
                    filename, rasa.shared.utils.io.DEFAULT_ENCODING
                ),
                add_line_numbers=add_line_numbers,
                file_path=filename,
            )
        except YamlException as e:
            e.filename = str(filename)
            raise e
        except RasaException as e:
            raise YamlException(filename) from e

    @staticmethod
    def humanize_flow_error(error: jsonschema.ValidationError) -> str:
        """Create a human understandable error message from a validation error.

        Converts a jsonschema validation error into a human understandable
        error message. This is used to provide more helpful error messages
        when a user provides an invalid flow definition.

        The documentation for the `jsonschema.ValidationError` can be found
        here https://python-jsonschema.readthedocs.io/en/latest/errors/#best-match-and-relevance

        Args:
            error: The validation error to convert.

        Returns:
            A human understandable error message.
        """

        def faulty_property(path: List[Any]) -> str:
            """Get the name of the property that caused the error.

            The exception contains a path to the property that caused the error.
            We will use that path to get the name of the property.

            Example:
                > faulty_property(['flows', 'add_contact', 'steps', 0, 'next'])
                'next'

            Args:
                path: The path to the property that caused the error.

            Returns:
                The name of the property that caused the error.
            """
            if not path:
                return "schema"
            if isinstance(path[-1], int):
                # the path is pointing towards an element in a list, so
                # we use the name of the list if possible
                return path[-2] if len(path) > 1 else "list"
            return str(path[-1])

        def schema_name(schema: Dict[str, Any]) -> str:
            """Get the name of the schema.

            This helps when displaying error messages, as we don't want to
            show the schema itself, but rather a name that describes
            what we expect. E.g. the following schema

            ```
            "set_slots": {
                "type": "array",
                "schema_name": "list of slot sets",
                "items": {
                    "type": "object"
                }
            }
            ```
            has a `schema_name` set. When we need to raise an error because
            this schema was not satisified, we will use the `schema_name`
            instead of the type itself. The type is less specific (`array`)
            and therefore less usefull than the handcrafted `schema_name`.

            If a schema does not have a `schema_name` set, we will use the
            `type` instead as a fallback.
            """
            return schema.get("schema_name", schema.get("type"))

        def schema_names(schemas: List[Dict[str, Any]]) -> List[str]:
            """Get the names of the schemas.

            Example:
                > schema_names([
                    {"required": ["action"], "schema_name": "action step"},
                    {"required": ["collect"], "schema_name": "collect step"},
                    {"required": ["link"], "schema_name": "link step"},
                    {"required": ["set_slots"], "schema_name": "slot set step"},
                    {"required": ["noop"], "schema_name": ""}])
                ['action step', 'collect step', 'link step', 'slot set step']

            Args:
                schemas: The schemas to get the names of.

            Returns:
                The names of the schemas.
            """
            names = []
            for schema in schemas:
                if name := schema_name(schema):
                    names.append(name)
            return names

        def expected_schema(error: jsonschema.ValidationError, schema_type: str) -> str:
            """Get the expected schema."""
            expected_schemas = error.schema.get(schema_type, [])
            expected = schema_names(expected_schemas)
            if expected:
                return " or ".join(sorted(expected))
            else:
                return str(error.schema)

        def format_oneof_error(error: jsonschema.ValidationError) -> str:
            """Format a oneOf error."""
            return (
                f"Not a valid '{faulty_property(error.absolute_path)}' definition. "
                f"Expected {expected_schema(error, 'oneOf')}."
            )

        def format_anyof_error(error: jsonschema.ValidationError) -> str:
            """Format an anyOf error."""
            return (
                f"Not a valid '{faulty_property(error.absolute_path)}' definition. "
                f"Expected {expected_schema(error, 'anyOf')}."
            )

        def format_type_error(error: jsonschema.ValidationError) -> str:
            """Format a type error."""
            expected_value = schema_name(error.schema)
            if isinstance(error.instance, dict):
                instance = "a dictionary"
            elif isinstance(error.instance, list):
                instance = "a list"
            else:
                instance = f"`{error.instance}`"
            return f"Found {instance} but expected a {expected_value}."

        if error.validator == "oneOf":
            return format_oneof_error(error)

        if error.validator == "anyOf":
            return format_anyof_error(error)

        if error.validator == "type":
            return format_type_error(error)

        if error.validator == "additionalProperties":
            return error.message

        if error.validator == "required":
            return error.message

        return (
            f"The flow at {error.json_path} is not valid. "
            f"Please double check your flow definition."
        )

    @classmethod
    def read_from_string(
        cls,
        string: Text,
        add_line_numbers: bool = True,
        file_path: Optional[Union[str, Path]] = None,
    ) -> FlowsList:
        """Read flows from a string.

        Args:
            string: Unprocessed YAML file content.
            add_line_numbers: If true, a custom constructor is added to add line
                numbers to each node.
            file_path: File path of the flow.

        Returns:
            `Flow`s read from `string`.
        """
        validate_yaml_with_jsonschema(
            string, FLOWS_SCHEMA_FILE, humanize_error=cls.humanize_flow_error
        )
        if add_line_numbers:
            yaml_content = read_yaml(string, custom_constructor=line_number_constructor)
            yaml_content = process_yaml_content(yaml_content)

        else:
            yaml_content = read_yaml(string)

        return FlowsList.from_json(yaml_content.get(KEY_FLOWS, {}), file_path=file_path)


class YamlFlowsWriter:
    """Class that writes flows information in YAML format."""

    @staticmethod
    def dumps(flows: List[Flow]) -> Text:
        """Dump `Flow`s to YAML.

        Args:
            flows: The `Flow`s to dump.

        Returns:
            The dumped YAML.
        """
        dump = {}
        for flow in flows:
            dumped_flow = flow.as_json()
            del dumped_flow["id"]
            dump[flow.id] = dumped_flow
        return dump_obj_as_yaml_to_string({KEY_FLOWS: dump})

    @staticmethod
    def dump(flows: List[Flow], filename: Union[Text, Path]) -> None:
        """Dump `Flow`s to YAML file.

        Args:
            flows: The `Flow`s to dump.
            filename: The path to the file to write to.
        """
        rasa.shared.utils.io.write_text_file(YamlFlowsWriter.dumps(flows), filename)


def flows_from_str(yaml_str: str) -> FlowsList:
    """Reads flows from a YAML string."""
    flows = YAMLFlowsReader.read_from_string(
        textwrap.dedent(yaml_str), add_line_numbers=False
    )
    flows.validate()
    return flows


def flows_from_str_including_defaults(yaml_str: str) -> FlowsList:
    """Reads flows from a YAML string and combine them with default flows."""
    flows = YAMLFlowsReader.read_from_string(
        textwrap.dedent(yaml_str), add_line_numbers=False
    )
    all_flows = FlowSyncImporter.merge_with_default_flows(flows)
    all_flows.validate()
    return all_flows


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
    return rasa.shared.data.is_likely_yaml_file(file_path) and is_key_in_yaml(
        file_path, KEY_FLOWS
    )


def line_number_constructor(loader: yaml.Loader, node: yaml_nodes.Node) -> Any:
    """A custom YAML constructor adding line numbers to nodes.

    Args:
        loader (yaml.Loader): The YAML loader.
        node (yaml.nodes.Node): The YAML node.

    Returns:
        Any: The constructed Python object with added line numbers in metadata.
    """
    if isinstance(node, yaml_nodes.MappingNode):
        mapping = loader.construct_mapping(node, deep=True)
        if "metadata" not in mapping:
            # We add the line information to the metadata of a flow step
            # Lines are 0-based index; adding 1 to start from
            # line 1 for human readability
            mapping["metadata"] = {
                "line_numbers": f"{node.start_mark.line + 1}-{node.end_mark.line}"
            }
        return mapping
    elif isinstance(node, yaml_nodes.SequenceNode):
        sequence = loader.construct_sequence(node, deep=True)
        for item in node.value:
            if isinstance(item, yaml_nodes.MappingNode):
                start_line = item.start_mark.line + 1
                end_line = item.end_mark.line
                # Only add line numbers to dictionary items within the sequence
                index = node.value.index(item)
                if isinstance(sequence[index], dict):
                    if "metadata" not in sequence[index]:
                        sequence[index]["metadata"] = {}
                    if "line_numbers" not in sequence[index]["metadata"]:
                        sequence[index]["metadata"]["line_numbers"] = (
                            f"{start_line}-{end_line}"
                        )

        return sequence
    return loader.construct_object(node, deep=True)


def _remove_keys_recursively(
    data: Union[Dict, List], keys_to_delete: List[str]
) -> None:
    """Recursively removes all specified keys in the given data.

    Special handling for 'metadata'.

    Args:
        data: The data structure (dictionary or list) to clean.
        keys_to_delete: A list of keys to remove from the dictionaries.
    """
    if isinstance(data, dict):
        keys = list(data.keys())
        for key in keys:
            if key in keys_to_delete:
                # Special case for 'metadata': only delete if it
                # only contains 'line_numbers'
                if key == "metadata" and isinstance(data[key], dict):
                    if len(data[key]) == 1 and "line_numbers" in data[key]:
                        del data[key]
                else:
                    del data[key]
            else:
                _remove_keys_recursively(data[key], keys_to_delete)
    elif isinstance(data, list):
        for item in data:
            _remove_keys_recursively(item, keys_to_delete)


def _process_keys_recursively(
    data: Union[Dict, List], keys_to_check: List[str]
) -> None:
    """Recursively iterates over YAML content and applies remove_keys_recursively."""
    if isinstance(data, dict):
        keys = list(
            data.keys()
        )  # Make a list of keys to avoid changing the dictionary size during iteration
        for key in keys:
            if key in keys_to_check:
                _remove_keys_recursively(data[key], ["metadata"])
            else:
                _process_keys_recursively(data[key], keys_to_check)
    elif isinstance(data, list):
        for item in data:
            _process_keys_recursively(item, keys_to_check)


def process_yaml_content(yaml_content: Dict[str, Any]) -> Dict[str, Any]:
    """Processes parsed YAML content to remove "metadata"."""
    # Remove metadata on the top level
    if "metadata" in yaml_content and (
        len(yaml_content["metadata"]) == 1
        and "line_numbers" in yaml_content["metadata"]
    ):
        del yaml_content["metadata"]

    # We expect metadata only under "flows" key...
    keys_to_delete_metadata = [key for key in yaml_content if key != "flows"]
    for key in keys_to_delete_metadata:
        _remove_keys_recursively(yaml_content[key], ["metadata"])

    # ...but "metadata" is also not a key of "flows"
    if "flows" in yaml_content and "metadata" in yaml_content["flows"]:
        if (
            len(yaml_content["flows"]["metadata"]) == 1
            and "line_numbers" in yaml_content["flows"]["metadata"]
        ):
            del yaml_content["flows"]["metadata"]

    # Under the "flows" key certain keys cannot have metadata
    _process_keys_recursively(
        yaml_content["flows"], ["nlu_trigger", "set_slots", "metadata"]
    )

    return yaml_content
