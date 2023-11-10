import textwrap
from pathlib import Path
from typing import List, Text, Union

import rasa.shared
import rasa.shared.data
import rasa.shared.utils.io
import rasa.shared.utils.validation
from rasa.shared.exceptions import YamlException

from rasa.shared.core.flows.flow import Flow
from rasa.shared.core.flows.flows_list import FlowsList

FLOWS_SCHEMA_FILE = "shared/core/flows/flows_yaml_schema.json"
KEY_FLOWS = "flows"


class YAMLFlowsReader:
    """Class that reads flows information in YAML format."""

    @classmethod
    def read_from_file(
        cls, filename: Union[Text, Path], skip_validation: bool = False
    ) -> FlowsList:
        """Read flows from file.

        Args:
            filename: Path to the flows file.
            skip_validation: `True` if the file was already validated
                e.g. when it was stored in the database.

        Returns:
            `Flow`s read from `filename`.
        """
        try:
            return cls.read_from_string(
                rasa.shared.utils.io.read_file(
                    filename, rasa.shared.utils.io.DEFAULT_ENCODING
                ),
                skip_validation,
            )
        except YamlException as e:
            e.filename = str(filename)
            raise e

    @classmethod
    def read_from_string(cls, string: Text, skip_validation: bool = False) -> FlowsList:
        """Read flows from a string.

        Args:
            string: Unprocessed YAML file content.
            skip_validation: `True` if the string was already validated
                e.g. when it was stored in the database.

        Returns:
            `Flow`s read from `string`.
        """
        if not skip_validation:
            rasa.shared.utils.validation.validate_yaml_with_jsonschema(
                string, FLOWS_SCHEMA_FILE
            )

        yaml_content = rasa.shared.utils.io.read_yaml(string)

        flows = FlowsList.from_json(yaml_content.get(KEY_FLOWS, {}))
        if not skip_validation:
            flows.validate()
        return flows


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
        return rasa.shared.utils.io.dump_obj_as_yaml_to_string({KEY_FLOWS: dump})

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
    return YAMLFlowsReader.read_from_string(textwrap.dedent(yaml_str))


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
    return rasa.shared.data.is_likely_yaml_file(
        file_path
    ) and rasa.shared.utils.io.is_key_in_yaml(file_path, KEY_FLOWS)
