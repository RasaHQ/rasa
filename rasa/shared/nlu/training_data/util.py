import json
import logging
import os
import re
from typing import Any, Dict, Optional, Text, Match

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
import rasa.shared.utils.io

logger = logging.getLogger(__name__)

ESCAPE_DCT = {"\b": "\\b", "\f": "\\f", "\n": "\\n", "\r": "\\r", "\t": "\\t"}
ESCAPE = re.compile(f'[{"".join(ESCAPE_DCT.values())}]')
UNESCAPE_DCT = {espaced_char: char for char, espaced_char in ESCAPE_DCT.items()}
UNESCAPE = re.compile(f'[{"".join(UNESCAPE_DCT.values())}]')
GROUP_COMPLETE_MATCH = 0


def transform_entity_synonyms(
    synonyms, known_synonyms: Optional[Dict[Text, Any]] = None
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


def get_file_format(resource_name: Text) -> Text:
    from rasa.shared.nlu.training_data import loading

    if resource_name is None or not os.path.exists(resource_name):
        raise AttributeError(f"Resource '{resource_name}' does not exist.")

    files = rasa.shared.utils.io.list_files(resource_name)

    file_formats = list(map(lambda f: loading.guess_format(f), files))

    if not file_formats:
        return "json"

    fformat = file_formats[0]
    if fformat in [loading.MARKDOWN, loading.RASA_YAML] and all(
        f == fformat for f in file_formats
    ):
        return fformat

    return "json"


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
