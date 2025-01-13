from typing import Any, Dict, List, Text

from rasa.shared.nlu.constants import (
    ENTITY_ATTRIBUTE_END,
    ENTITY_ATTRIBUTE_START,
    ENTITY_ATTRIBUTE_VALUE,
)


def add_synonyms_from_entities(
    plain_text: Text, entities: List[Dict], existing_synonyms: Dict[Text, Any]
) -> None:
    """Adds synonyms found in intent examples.

    Args:
        plain_text: Plain (with removed special symbols) user utterance.
        entities: Entities that were extracted from the original user utterance.
        existing_synonyms: The dict with existing synonyms mappings that will
                           be extended.
    """
    for e in entities:
        e_text = plain_text[e[ENTITY_ATTRIBUTE_START] : e[ENTITY_ATTRIBUTE_END]]
        if e_text != e[ENTITY_ATTRIBUTE_VALUE]:
            add_synonym(e_text, e[ENTITY_ATTRIBUTE_VALUE], existing_synonyms)


def add_synonym(
    synonym_value: Text, synonym_name: Text, existing_synonyms: Dict[Text, Any]
) -> None:
    """Adds a new synonym mapping to the provided list of synonyms.

    Args:
        synonym_value: Value of the synonym.
        synonym_name: Name of the synonym.
        existing_synonyms: Dictionary will synonym mappings that will be extended.
    """
    import rasa.shared.nlu.training_data.util as training_data_util

    training_data_util.check_duplicate_synonym(
        existing_synonyms, synonym_value, synonym_name, "reading markdown"
    )
    existing_synonyms[synonym_value] = synonym_name
