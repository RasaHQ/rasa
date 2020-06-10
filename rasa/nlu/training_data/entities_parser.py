# regex for: `[entity_text]((entity_type(:entity_synonym)?)|{entity_dict})`
import re
from json import JSONDecodeError
from typing import Text, List, Dict, Match, Optional, NamedTuple

from rasa.nlu.utils import build_entity

from rasa.constants import DOCS_URL_TRAINING_DATA_NLU

from rasa.nlu.constants import (
    ENTITY_ATTRIBUTE_GROUP,
    ENTITY_ATTRIBUTE_TYPE,
    ENTITY_ATTRIBUTE_ROLE,
    ENTITY_ATTRIBUTE_VALUE,
)

from rasa.utils.common import raise_warning

GROUP_ENTITY_VALUE = "value"
GROUP_ENTITY_TYPE = "entity"
GROUP_ENTITY_DICT = "entity_dict"
GROUP_ENTITY_TEXT = "entity_text"
GROUP_COMPLETE_MATCH = 0

# regex for: `[entity_text]((entity_type(:entity_synonym)?)|{entity_dict})`
ENTITY_REGEX = re.compile(
    r"\[(?P<entity_text>[^\]]+?)\](\((?P<entity>[^:)]+?)(?:\:(?P<value>[^)]+))?\)|\{(?P<entity_dict>[^}]+?)\})"
)


class EntityAttributes(NamedTuple):
    """Attributes of an entity defined in the markdown data."""

    type: Text
    value: Text
    text: Text
    group: Optional[Text]
    role: Optional[Text]


class EntitiesParser:
    @staticmethod
    def find_entities_in_training_example(example: Text) -> List[Dict]:
        """Extracts entities from an intent example.

        Args:
            example: intent example

        Returns: list of extracted entities
        """
        entities = []
        offset = 0

        for match in re.finditer(ENTITY_REGEX, example):
            entity_attributes = EntitiesParser._extract_entity_attributes(match)

            start_index = match.start() - offset
            end_index = start_index + len(entity_attributes.text)
            offset += len(match.group(0)) - len(entity_attributes.text)

            entity = build_entity(
                start_index,
                end_index,
                entity_attributes.value,
                entity_attributes.type,
                entity_attributes.role,
                entity_attributes.group,
            )
            entities.append(entity)

        return entities

    @staticmethod
    def _extract_entity_attributes(match: Match) -> EntityAttributes:
        """Extract the entity attributes, i.e. type, value, etc., from the
        regex match."""
        entity_text = match.groupdict()[GROUP_ENTITY_TEXT]

        if match.groupdict()[GROUP_ENTITY_DICT]:
            return EntitiesParser._extract_entity_attributes_from_dict(
                entity_text, match
            )

        entity_type = match.groupdict()[GROUP_ENTITY_TYPE]

        if match.groupdict()[GROUP_ENTITY_VALUE]:
            entity_value = match.groupdict()[GROUP_ENTITY_VALUE]
        else:
            entity_value = entity_text

        return EntityAttributes(entity_type, entity_value, entity_text, None, None)

    @staticmethod
    def _extract_entity_attributes_from_dict(
        entity_text: Text, match: Match
    ) -> EntityAttributes:
        """Extract the entity attributes from the dict format."""
        entity_dict_str = match.groupdict()[GROUP_ENTITY_DICT]
        entity_dict = EntitiesParser._get_validated_dict(entity_dict_str)
        return EntityAttributes(
            entity_dict.get(ENTITY_ATTRIBUTE_TYPE),
            entity_dict.get(ENTITY_ATTRIBUTE_VALUE, entity_text),
            entity_text,
            entity_dict.get(ENTITY_ATTRIBUTE_GROUP),
            entity_dict.get(ENTITY_ATTRIBUTE_ROLE),
        )

    @staticmethod
    def _get_validated_dict(json_str: Text) -> Dict[Text, Text]:
        """Converts the provided json_str to a valid dict containing the entity
        attributes.

        Users can specify entity roles, synonyms, groups for an entity in a dict, e.g.
        [LA]{"entity": "city", "role": "to", "value": "Los Angeles"}

        Args:
            json_str: the entity dict as string without "{}"

        Raises:
            ValidationError if validation of entity dict fails.
            JSONDecodeError if provided entity dict is not valid json.

        Returns:
            a proper python dict
        """
        import json
        import rasa.utils.validation as validation_utils
        import rasa.nlu.schemas.data_schema as schema

        # add {} as they are not part of the regex
        try:
            data = json.loads(f"{{{json_str}}}")
        except JSONDecodeError as e:
            raise_warning(
                f"Incorrect training data format ('{{{json_str}}}'), make sure your "
                f"data is valid. For more information about the format visit "
                f"{DOCS_URL_TRAINING_DATA_NLU}."
            )
            raise e

        validation_utils.validate_training_data(data, schema.entity_dict_schema())

        return data

    @staticmethod
    def replace_entities(training_example: Text) -> Text:
        return re.sub(
            ENTITY_REGEX, lambda m: m.groupdict()[GROUP_ENTITY_TEXT], training_example
        )
