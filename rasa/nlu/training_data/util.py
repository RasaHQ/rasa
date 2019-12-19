import logging
import os
import warnings
from typing import Any, Dict, Optional, Text

import rasa.utils.io as io_utils
from rasa.nlu.constants import ENTITIES_ATTRIBUTE, EXTRACTOR_ATTRIBUTE

logger = logging.getLogger(__name__)


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
        warnings.warn(
            f"Found inconsistent entity synonyms while {context_str}, "
            f"overwriting {text}->{entity_synonyms[text]} "
            f"with {text}->{syn} during merge."
        )


def get_file_format(resource_name: Text) -> Text:
    from rasa.nlu.training_data import loading

    if resource_name is None or not os.path.exists(resource_name):
        raise AttributeError(f"Resource '{resource_name}' does not exist.")

    files = io_utils.list_files(resource_name)

    file_formats = list(map(lambda f: loading.guess_format(f), files))

    if not file_formats:
        return "json"

    fformat = file_formats[0]
    if fformat == "md" and all(f == fformat for f in file_formats):
        return fformat

    return "json"


def remove_untrainable_entities_from(example: Dict[Text, Any]) -> None:
    """Remove untrainable entities from serialised training example `example`.

    Entities with an untrainable extractor will be removed. Untrainable extractors
    are `DucklingHTTPExtractor`, `MitieEntityExtractor` and `SpacyEntityExtractor`.

    Args:
        example: Serialised training example to inspect.
    """

    from rasa.nlu.extractors.duckling_http_extractor import DucklingHTTPExtractor
    from rasa.nlu.extractors.mitie_entity_extractor import MitieEntityExtractor
    from rasa.nlu.extractors.spacy_entity_extractor import SpacyEntityExtractor

    untrainable_entity_extractors = [
        DucklingHTTPExtractor.__name__,
        SpacyEntityExtractor.__name__,
        MitieEntityExtractor.__name__,
    ]

    example_entities = example.get(ENTITIES_ATTRIBUTE)
    if example_entities:
        trainable_entities = []
        for entity in example_entities:
            if entity.get(EXTRACTOR_ATTRIBUTE) in untrainable_entity_extractors:
                logger.debug(
                    f"Exclude entity marking of the following extractors "
                    f"{untrainable_entity_extractors} when writing NLU data "
                    f"to file."
                )
            else:
                trainable_entities.append(entity)

        example[ENTITIES_ATTRIBUTE] = trainable_entities
