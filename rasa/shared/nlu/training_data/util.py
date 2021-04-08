import json
import logging
import os
import re
from typing import Any, Dict, Optional, Text, Match, List

from rasa.shared.nlu.constants import (
    ENTITIES,
    EXTRACTOR,
    PRETRAINED_EXTRACTORS,
    ENTITY_ATTRIBUTE_START,
    ENTITY_ATTRIBUTE_END,
    ENTITY_ATTRIBUTE_VALUE,
    ENTITY_ATTRIBUTE_TYPE,
    ENTITY_ATTRIBUTE_ROLE,
    ENTITY_ATTRIBUTE_GROUP,
)
from rasa.shared.constants import UTTER_PREFIX
import rasa.shared.utils.io
import rasa.shared.data

logger = logging.getLogger(__name__)

ESCAPE_DCT = {"\b": "\\b", "\f": "\\f", "\n": "\\n", "\r": "\\r", "\t": "\\t"}
ESCAPE_CHARS = set(ESCAPE_DCT.keys())
ESCAPE = re.compile(f'[{"".join(ESCAPE_DCT.values())}]')
UNESCAPE_DCT = {espaced_char: char for char, espaced_char in ESCAPE_DCT.items()}
UNESCAPE = re.compile(f'[{"".join(UNESCAPE_DCT.values())}]')
GROUP_COMPLETE_MATCH = 0


def transform_entity_synonyms(
    synonyms: List[Dict[Text, Any]], known_synonyms: Optional[Dict[Text, Any]] = None
) -> Dict[Text, Any]:
    """Transforms the entity synonyms into a text->value dictionary"""
    entity_synonyms = known_synonyms if known_synonyms else {}
    for s in synonyms:
        if "value" in s and "synonyms" in s:
            for synonym in s["synonyms"]:
                entity_synonyms[synonym] = s["value"]
    return entity_synonyms


def check_duplicate_synonym(
    entity_synonyms: Dict[Text, Any], text: Text, syn: Text, context_str: Text = ""
) -> None:
    if text in entity_synonyms and entity_synonyms[text] != syn:
        rasa.shared.utils.io.raise_warning(
            f"Found inconsistent entity synonyms while {context_str}, "
            f"overwriting {text}->{entity_synonyms[text]} "
            f"with {text}->{syn} during merge."
        )


def get_file_format_extension(resource_name: Text) -> Text:
    """
    Get the file extension based on training data format. It supports both a folder and
    a file, and tries to guess the format as follows:

    - if the resource is a file and has a known format, return this format's extension
    - if the resource is a folder and all the resources have the
      same known format, return it's extension
    - otherwise, default to DEFAULT_FILE_FORMAT (yml).

    Args:
        resource_name: The name of the resource, can be a file or a folder.
    Returns:
        The resource file format.
    """
    from rasa.shared.nlu.training_data import loading

    if resource_name is None or not os.path.exists(resource_name):
        raise AttributeError(f"Resource '{resource_name}' does not exist.")

    files = rasa.shared.utils.io.list_files(resource_name)

    file_formats = list(map(lambda f: loading.guess_format(f), files))

    if not file_formats:
        return rasa.shared.data.yaml_file_extension()

    known_file_formats = {
        loading.MARKDOWN: rasa.shared.data.markdown_file_extension(),
        loading.RASA_YAML: rasa.shared.data.yaml_file_extension(),
    }
    fformat = file_formats[0]
    if all(f == fformat for f in file_formats):
        return known_file_formats.get(fformat, rasa.shared.data.yaml_file_extension())

    return rasa.shared.data.yaml_file_extension()


def remove_untrainable_entities_from(example: Dict[Text, Any]) -> None:
    """Remove untrainable entities from serialised training example `example`.

    Entities with an untrainable extractor will be removed. Untrainable extractors
    are defined in `rasa.nlu.constants.PRETRAINED_EXTRACTORS`.

    Args:
        example: Serialised training example to inspect.
    """

    example_entities = example.get(ENTITIES)

    if not example_entities:
        # example contains no entities, so there's nothing to do
        return None

    trainable_entities = []

    for entity in example_entities:
        if entity.get(EXTRACTOR) in PRETRAINED_EXTRACTORS:
            logger.debug(
                f"Excluding entity '{json.dumps(entity)}' from training data. "
                f"Entity examples extracted by the following classes are not "
                f"dumped to training data in markdown format: "
                f"`{'`, `'.join(sorted(PRETRAINED_EXTRACTORS))}`."
            )
        else:
            trainable_entities.append(entity)

    example[ENTITIES] = trainable_entities


def intent_response_key_to_template_key(intent_response_key: Text) -> Text:
    """Resolve the response template key for a given intent response key.

    Args:
        intent_response_key: retrieval intent with the response key suffix attached.

    Returns: The corresponding response template.

    """
    return f"{UTTER_PREFIX}{intent_response_key}"


def template_key_to_intent_response_key(template_key: Text) -> Text:
    """Resolve the intent response key for the given response template.

    Args:
        template_key: Name of the response template.

    Returns: The corresponding intent response key.

    """
    return template_key.split(UTTER_PREFIX)[1]


def has_string_escape_chars(s: Text) -> bool:
    """Checks whether there are any of the escape characters in the string."""
    intersection = ESCAPE_CHARS.intersection(set(s))
    return len(intersection) > 0


def encode_string(s: Text) -> Text:
    """Return an encoded python string."""

    def replace(match: Match) -> Text:
        return ESCAPE_DCT[match.group(GROUP_COMPLETE_MATCH)]

    return ESCAPE.sub(replace, s)


def decode_string(s: Text) -> Text:
    """Return a decoded python string."""

    def replace(match: Match) -> Text:
        return UNESCAPE_DCT[match.group(GROUP_COMPLETE_MATCH)]

    return UNESCAPE.sub(replace, s)


def build_entity(
    start: int,
    end: int,
    value: Text,
    entity_type: Text,
    role: Optional[Text] = None,
    group: Optional[Text] = None,
    **kwargs: Any,
) -> Dict[Text, Any]:
    """Builds a standard entity dictionary.

    Adds additional keyword parameters.

    Args:
        start: start position of entity
        end: end position of entity
        value: text value of the entity
        entity_type: name of the entity type
        role: role of the entity
        group: group of the entity
        **kwargs: additional parameters

    Returns:
        an entity dictionary
    """

    entity = {
        ENTITY_ATTRIBUTE_START: start,
        ENTITY_ATTRIBUTE_END: end,
        ENTITY_ATTRIBUTE_VALUE: value,
        ENTITY_ATTRIBUTE_TYPE: entity_type,
    }

    if role:
        entity[ENTITY_ATTRIBUTE_ROLE] = role
    if group:
        entity[ENTITY_ATTRIBUTE_GROUP] = group

    entity.update(kwargs)
    return entity
