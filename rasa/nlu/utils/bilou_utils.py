from typing import List, Tuple, Text, Optional, Dict, Set, Any

from rasa.nlu.tokenizers.tokenizer import Token
from rasa.nlu.training_data import Message
from rasa.nlu.training_data import TrainingData
from rasa.nlu.constants import (
    ENTITIES,
    TOKENS_NAMES,
    TEXT,
    BILOU_ENTITIES,
    NO_ENTITY_TAG,
    ENTITY_ATTRIBUTE_TYPE,
    ENTITY_ATTRIBUTE_END,
    ENTITY_ATTRIBUTE_START,
    BILOU_ENTITIES_GROUP,
    BILOU_ENTITIES_ROLE,
    ENTITY_ATTRIBUTE_ROLE,
    ENTITY_ATTRIBUTE_GROUP,
)

BILOU_PREFIXES = ["B-", "I-", "U-", "L-"]


def bilou_prefix_from_tag(tag: Text) -> Optional[Text]:
    """Returns the BILOU prefix from the given tag.

    Args:
        tag: the tag

    Returns: the BILOU prefix of the tag
    """
    if tag[:2] in BILOU_PREFIXES:
        return tag[0]
    return None


def entity_name_from_tag(tag: Text) -> Text:
    """Remove the BILOU prefix from the given tag.

    Args:
        tag: the tag

    Returns: the tag without the BILOU prefix
    """
    if tag[:2] in BILOU_PREFIXES:
        return tag[2:]
    return tag


def bilou_tags_to_ids(
    message: Message,
    tag_id_dict: Dict[Text, int],
    tag_name: Text = ENTITY_ATTRIBUTE_TYPE,
) -> List[int]:
    """Maps the entity tags of the message to the ids of the provided dict.

    Args:
        message: the message
        tag_id_dict: mapping of tags to ids
        tag_name: tag name of interest

    Returns: a list of tag ids
    """
    bilou_key = _get_bilou_key_for_tag(tag_name)

    if message.get(bilou_key):
        _tags = [
            tag_id_dict[_tag] if _tag in tag_id_dict else tag_id_dict[NO_ENTITY_TAG]
            for _tag in message.get(bilou_key)
        ]
    else:
        _tags = [tag_id_dict[NO_ENTITY_TAG] for _ in message.get(TOKENS_NAMES[TEXT])]

    return _tags


def _get_bilou_key_for_tag(tag_name: Text) -> Text:
    """Get the message key for the BILOU tagging format of the provided tag name.

    Args:
        tag_name: the tag name

    Returns:
        the message key to store the BILOU tags
    """
    if tag_name == ENTITY_ATTRIBUTE_ROLE:
        return BILOU_ENTITIES_ROLE

    if tag_name == ENTITY_ATTRIBUTE_GROUP:
        return BILOU_ENTITIES_GROUP

    return BILOU_ENTITIES


def remove_bilou_prefixes(tags: List[Text]) -> List[Text]:
    """Removes the BILOU prefixes from the given list of tags.

    Args:
        tags: the list of tags

    Returns:
        list of tags without BILOU prefix
    """
    return [entity_name_from_tag(t) for t in tags]


def build_tag_id_dict(
    training_data: TrainingData, tag_name: Text = ENTITY_ATTRIBUTE_TYPE
) -> Optional[Dict[Text, int]]:
    """Create a mapping of unique tags to ids.

    Args:
        training_data: the training data
        tag_name: tag name of interest

    Returns: a mapping of tags to ids
    """
    bilou_key = _get_bilou_key_for_tag(tag_name)

    distinct_tags = set(
        [
            entity_name_from_tag(e)
            for example in training_data.training_examples
            if example.get(bilou_key)
            for e in example.get(bilou_key)
        ]
    ) - {NO_ENTITY_TAG}

    if not distinct_tags:
        return None

    tag_id_dict = {
        f"{prefix}{tag}": idx_1 * len(BILOU_PREFIXES) + idx_2 + 1
        for idx_1, tag in enumerate(sorted(distinct_tags))
        for idx_2, prefix in enumerate(BILOU_PREFIXES)
    }
    # NO_ENTITY_TAG corresponds to non-entity which should correspond to 0 index
    # needed for correct prediction for padding
    tag_id_dict[NO_ENTITY_TAG] = 0

    return tag_id_dict


def apply_bilou_schema(training_data: TrainingData) -> None:
    """Gets a list of BILOU entity tags and sets them on the given messages.

    Args:
        training_data: the training data
    """
    for message in training_data.training_examples:
        entities = message.get(ENTITIES)

        if not entities:
            continue

        for attribute, message_key in [
            (ENTITY_ATTRIBUTE_TYPE, BILOU_ENTITIES),
            (ENTITY_ATTRIBUTE_ROLE, BILOU_ENTITIES_ROLE),
            (ENTITY_ATTRIBUTE_GROUP, BILOU_ENTITIES_GROUP),
        ]:
            entities = map_message_entities(message, attribute)
            output = bilou_tags_from_offsets(message.get(TOKENS_NAMES[TEXT]), entities)

            message.set(message_key, output)


def map_message_entities(
    message: Message, attribute_key: Text = ENTITY_ATTRIBUTE_TYPE
) -> List[Tuple[int, int, Text]]:
    """Maps the entities of the given message to their start, end, and tag values.

    Args:
        message: the message
        attribute_key: key of tag value to use

    Returns: a list of start, end, and tag value tuples
    """

    def convert_entity(entity: Dict[Text, Any]) -> Tuple[int, int, Text]:
        return (
            entity[ENTITY_ATTRIBUTE_START],
            entity[ENTITY_ATTRIBUTE_END],
            entity.get(attribute_key) or NO_ENTITY_TAG,
        )

    return [convert_entity(entity) for entity in message.get(ENTITIES, [])]


def bilou_tags_from_offsets(
    tokens: List[Token],
    entities: List[Tuple[int, int, Text]],
    missing: Text = NO_ENTITY_TAG,
) -> List[Text]:
    """Creates a list of BILOU tags for the given list of tokens and entities.

    Args:
        tokens: the list of tokens
        entities: the list of start, end, and tag tuples
        missing: tag for missing entities

    Returns: a list of BILOU tags
    """
    # From spacy.spacy.GoldParse, under MIT License

    start_pos_to_token_idx = {token.start: i for i, token in enumerate(tokens)}
    end_pos_to_token_idx = {token.end: i for i, token in enumerate(tokens)}

    bilou = ["-" for _ in tokens]

    # Handle entity cases
    _add_bilou_tags_to_entities(
        bilou, entities, end_pos_to_token_idx, start_pos_to_token_idx
    )

    # Now distinguish the O cases from ones where we miss the tokenization
    entity_positions = _get_entity_positions(entities)
    _handle_not_an_entity(bilou, tokens, entity_positions, missing)

    return bilou


def _add_bilou_tags_to_entities(
    bilou: List[Text],
    entities: List[Tuple[int, int, Text]],
    end_pos_to_token_idx: Dict[int, int],
    start_pos_to_token_idx: Dict[int, int],
):
    for start_pos, end_pos, label in entities:
        start_token_idx = start_pos_to_token_idx.get(start_pos)
        end_token_idx = end_pos_to_token_idx.get(end_pos)

        # Only interested if the tokenization is correct
        if start_token_idx is not None and end_token_idx is not None:
            if start_token_idx == end_token_idx:
                bilou[start_token_idx] = f"U-{label}"
            else:
                bilou[start_token_idx] = f"B-{label}"
                for i in range(start_token_idx + 1, end_token_idx):
                    bilou[i] = f"I-{label}"
                bilou[end_token_idx] = f"L-{label}"


def _get_entity_positions(entities: List[Tuple[int, int, Text]]) -> Set[int]:
    entity_positions = set()

    for start_pos, end_pos, label in entities:
        for i in range(start_pos, end_pos):
            entity_positions.add(i)

    return entity_positions


def _handle_not_an_entity(
    bilou: List[Text], tokens: List[Token], entity_positions: Set[int], missing: Text
):
    for n, token in enumerate(tokens):
        for i in range(token.start, token.end):
            if i in entity_positions:
                break
        else:
            bilou[n] = missing
