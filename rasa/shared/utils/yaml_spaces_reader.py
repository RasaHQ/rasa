from typing import Union, Text, List, Optional, Dict
from pathlib import Path
from dataclasses import dataclass
import rasa.shared.utils.io
import rasa.shared.utils.validation

from rasa.shared.exceptions import YamlException

KEY_SPACES = "spaces"
KEY_INCLUDE_AT = "include_at"
KEY_DOMAIN = "domain"
KEY_NLU = "nlu"
KEY_STORIES = "stories"
KEY_ENTRY_INTENTS = "entry_intents"
KEY_ACTIVE_AT_SESSION_START = "active_at_session_start"

SPACES_SCHEMA_FILE = "shared/utils/schemas/spaces.yml"


@dataclass
class Space:
    name: Text
    domain_location: Text
    entry_intents: List[Text]
    active_at_session_start: bool
    nlu_location: Optional[Text]
    stories_location: Optional[Text]


class YAMLSpacesReader:
    """Class that reads spaces information in YAML format."""

    @classmethod
    def read_from_file(
        cls, filename: Union[Text, Path], skip_validation: bool = False
    ) -> List[Space]:
        """Read spaces from file.

        Args:
            filename: Path to the spaces file.
            skip_validation: `True` if the file was already validated
                e.g. when it was stored in the database.

        Returns:
            `Space`s read from `filename`.
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
    def read_from_string(
        cls, string: Text, skip_validation: bool = False
    ) -> List[Space]:
        """Read spaces from a string.

        Args:
            string: Unprocessed YAML file content.
            skip_validation: `True` if the string was already validated
                e.g. when it was stored in the database.

        Returns:
            `Space`s read from `string`.
        """
        if not skip_validation:
            rasa.shared.utils.validation.validate_yaml_schema(string, SPACES_SCHEMA_FILE)

        yaml_content = rasa.shared.utils.io.read_yaml(string)

        return cls.read_from_parsed_yaml(yaml_content)

    @staticmethod
    def read_from_parsed_yaml(
        parsed_content: Dict[Text, Union[Dict, List]]
    ) -> List[Space]:
        """Read spaces from parsed YAML.

        Args:
            parsed_content: The parsed YAML as a dictionary.

        Returns:
            The parsed spaces.
        """
        spaces = []
        for space_content in parsed_content[KEY_SPACES]:
            space = Space(
                space_content[KEY_INCLUDE_AT],
                space_content[KEY_DOMAIN],
                space_content.get(KEY_ENTRY_INTENTS, []),
                space_content.get(KEY_ACTIVE_AT_SESSION_START, False),
                space_content.get(KEY_NLU),
                space_content.get(KEY_STORIES)
            )
            spaces.append(space)
        return spaces
