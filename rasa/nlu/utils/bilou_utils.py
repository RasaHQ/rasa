import logging
from typing import List, Tuple, Text, Optional, Dict, Any

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
import rasa.utils.train_utils as train_utils

logger = logging.getLogger(__name__)

BEGINNING = "B-"
INSIDE = "I-"
LAST = "L-"
UNIT = "U-"
BILOU_PREFIXES = [BEGINNING, INSIDE, LAST, UNIT]


def bilou_prefix_from_tag(tag: Text) -> Optional[Text]:
    """Returns the BILOU prefix from the given tag.

    Args:
        tag: the tag

    Returns: the BILOU prefix of the tag
    """
    if tag[:2] in BILOU_PREFIXES:
        return tag[:2]
    return None


def tag_without_prefix(tag: Text) -> Text:
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
    bilou_key = get_bilou_key_for_tag(tag_name)

    if message.get(bilou_key):
        _tags = [
            tag_id_dict[_tag] if _tag in tag_id_dict else tag_id_dict[NO_ENTITY_TAG]
            for _tag in message.get(bilou_key)
        ]
    else:
        _tags = [tag_id_dict[NO_ENTITY_TAG] for _ in message.get(TOKENS_NAMES[TEXT])]

    return _tags


def get_bilou_key_for_tag(tag_name: Text) -> Text:
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
    return [tag_without_prefix(t) for t in tags]


def build_tag_id_dict(
    training_data: TrainingData, tag_name: Text = ENTITY_ATTRIBUTE_TYPE
) -> Optional[Dict[Text, int]]:
    """Create a mapping of unique tags to ids.

    Args:
        training_data: the training data
        tag_name: tag name of interest

    Returns: a mapping of tags to ids
    """
    bilou_key = get_bilou_key_for_tag(tag_name)

    distinct_tags = set(
        [
            tag_without_prefix(e)
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


def apply_bilou_schema(
    training_data: TrainingData, include_cls_token: bool = True
) -> None:
    """Get a list of BILOU entity tags and set them on the given messages.

    Args:
        training_data: the training data
    """
    for message in training_data.training_examples:
        entities = message.get(ENTITIES)

        if not entities:
            continue

        tokens = message.get(TOKENS_NAMES[TEXT])
        if not include_cls_token:
            tokens = train_utils.tokens_without_cls(message)

        for attribute, message_key in [
            (ENTITY_ATTRIBUTE_TYPE, BILOU_ENTITIES),
            (ENTITY_ATTRIBUTE_ROLE, BILOU_ENTITIES_ROLE),
            (ENTITY_ATTRIBUTE_GROUP, BILOU_ENTITIES_GROUP),
        ]:
            entities = map_message_entities(message, attribute)
            output = bilou_tags_from_offsets(tokens, entities)
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

    entities = [convert_entity(entity) for entity in message.get(ENTITIES, [])]

    # entities is a list of tuples (start, end, tag value).
    # filter out all entities with tag value == NO_ENTITY_TAG.
    tag_value_idx = 2
    return [entity for entity in entities if entity[tag_value_idx] != NO_ENTITY_TAG]


def bilou_tags_from_offsets(
    tokens: List[Token], entities: List[Tuple[int, int, Text]]
) -> List[Text]:
    """Creates BILOU tags for the given tokens and entities.

    Args:
        message: The message object.
        tokens: The list of tokens.
        entities: The list of start, end, and tag tuples.
        missing: The tag for missing entities.

    Returns:
        BILOU tags.
    """
    start_pos_to_token_idx = {token.start: i for i, token in enumerate(tokens)}
    end_pos_to_token_idx = {token.end: i for i, token in enumerate(tokens)}

    bilou = [NO_ENTITY_TAG for _ in tokens]

    _add_bilou_tags_to_entities(
        bilou, entities, end_pos_to_token_idx, start_pos_to_token_idx
    )

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
                bilou[start_token_idx] = f"{UNIT}{label}"
            else:
                bilou[start_token_idx] = f"{BEGINNING}{label}"
                for i in range(start_token_idx + 1, end_token_idx):
                    bilou[i] = f"{INSIDE}{label}"
                bilou[end_token_idx] = f"{LAST}{label}"


def ensure_consistent_bilou_tagging(predicted_tags: List[Text]) -> List[Text]:
    """
    Ensure predicted tags follow the BILOU tagging schema.

    We assume that starting B- tags are correct. Followed tags that belong to start
    tag but have a different entity type are updated.
    For example, B-a I-b L-a is updated to B-a I-a L-a and B-a I-a O is changed to
    B-a L-a.

    Args:
        predicted_tags: predicted tags

    Return:
        List of tags.
    """

    for idx, predicted_tag in enumerate(predicted_tags):
        prefix = bilou_prefix_from_tag(predicted_tag)
        tag = tag_without_prefix(predicted_tag)

        if prefix == BEGINNING:
            last_idx = _find_bilou_end(idx, predicted_tags)

            # ensure correct BILOU annotations
            if last_idx == idx:
                predicted_tags[idx] = f"{UNIT}{tag}"
            elif last_idx - idx == 1:
                predicted_tags[idx] = f"{BEGINNING}{tag}"
                predicted_tags[last_idx] = f"{LAST}{tag}"
            else:
                predicted_tags[idx] = f"{BEGINNING}{tag}"
                predicted_tags[last_idx] = f"{LAST}{tag}"
                for i in range(idx + 1, last_idx):
                    predicted_tags[i] = f"{INSIDE}{tag}"

    return predicted_tags


def _find_bilou_end(start_idx: int, predicted_tags: List[Text]) -> int:
    current_idx = start_idx + 1
    finished = False
    start_tag = tag_without_prefix(predicted_tags[start_idx])

    while not finished:
        if current_idx >= len(predicted_tags):
            logger.debug(
                "Inconsistent BILOU tagging found, B- tag not closed by L- tag, "
                "i.e [B-a, I-a, O] instead of [B-a, L-a, O].\n"
                "Assuming last tag is L- instead of I-."
            )
            current_idx -= 1
            break

        current_label = predicted_tags[current_idx]
        prefix = bilou_prefix_from_tag(current_label)
        tag = tag_without_prefix(current_label)

        if tag != start_tag:
            # words are not tagged the same entity class
            logger.debug(
                "Inconsistent BILOU tagging found, B- tag, L- tag pair encloses "
                "multiple entity classes.i.e. [B-a, I-b, L-a] instead of "
                "[B-a, I-a, L-a].\nAssuming B- class is correct."
            )

        if prefix == LAST:
            finished = True
        elif prefix == INSIDE:
            # middle part of the entity
            current_idx += 1
        else:
            # entity not closed by an L- tag
            finished = True
            current_idx -= 1
            logger.debug(
                "Inconsistent BILOU tagging found, B- tag not closed by L- tag, "
                "i.e [B-a, I-a, O] instead of [B-a, L-a, O].\n"
                "Assuming last tag is L- instead of I-."
            )

    return current_idx
