from typing import List, Tuple, Text, Optional, Dict, Set

from rasa.nlu.tokenizers.tokenizer import Token
from rasa.nlu.training_data import Message
from rasa.nlu.training_data import TrainingData
from rasa.nlu.constants import (
    ENTITIES,
    TOKENS_NAMES,
    TEXT,
    BILOU_ENTITIES,
)

BILOU_PREFIXES = ["B-", "I-", "U-", "L-"]


def entity_name_from_tag(tag: Text) -> Text:
    """Remove the BILOU prefix from the given tag."""
    if tag[:2] in BILOU_PREFIXES:
        return tag[2:]
    return tag


def tags_to_ids(message: Message, tag_id_dict: Dict[Text, int]) -> List[int]:
    """Maps the entity tags of the message to the ids of the provided dict."""
    if message.get(BILOU_ENTITIES):
        _tags = [
            tag_id_dict[_tag] if _tag in tag_id_dict else tag_id_dict["O"]
            for _tag in message.get(BILOU_ENTITIES)
        ]
    else:
        _tags = [tag_id_dict["O"] for _ in message.get(TOKENS_NAMES[TEXT])]

    return _tags


def remove_bilou_prefixes(tags: List[Text]) -> List[Text]:
    """Remove the BILOU prefixes from the given tags."""
    return [entity_name_from_tag(t) for t in tags]


def build_tag_id_dict(training_data: TrainingData) -> Dict[Text, int]:
    """Create a mapping of unique tags to ids."""
    distinct_tags = set(
        [
            entity_name_from_tag(e)
            for example in training_data.training_examples
            if example.get(BILOU_ENTITIES)
            for e in example.get(BILOU_ENTITIES)
        ]
    ) - {"O"}

    tag_id_dict = {
        f"{prefix}{tag}": idx_1 * len(BILOU_PREFIXES) + idx_2 + 1
        for idx_1, tag in enumerate(sorted(distinct_tags))
        for idx_2, prefix in enumerate(BILOU_PREFIXES)
    }
    tag_id_dict["O"] = 0

    return tag_id_dict


def apply_bilou_schema(training_data: TrainingData):
    """Obtains a list of BILOU entity tags and sets them on the corresponding
    message."""
    for message in training_data.training_examples:
        entities = message.get(ENTITIES)

        if not entities:
            continue

        entities = _map_message_entities(message)
        output = _bilou_tags_from_offsets(message.get(TOKENS_NAMES[TEXT]), entities)

        message.set(BILOU_ENTITIES, output)


def _map_message_entities(message: Message) -> List[Tuple[int, int, Text]]:
    """Maps the entities of the given message to their start, end, and tag values."""

    def convert_entity(entity):
        return entity["start"], entity["end"], entity["entity"]

    return [convert_entity(entity) for entity in message.get(ENTITIES, [])]


def _bilou_tags_from_offsets(
    tokens: List[Token], entities: List[Tuple[int, int, Text]], missing: Text = "O"
) -> List[Text]:
    """Creates a list of BILOU tags for the given list of tokens and entities."""

    # From spacy.spacy.GoldParse, under MIT License

    start_pos_to_token_idx = {token.start: i for i, token in enumerate(tokens)}
    end_pos_to_token_idx = {token.end: i for i, token in enumerate(tokens)}

    bilou = ["-" for _ in tokens]

    # Handle entity cases
    _handle_entities(bilou, entities, end_pos_to_token_idx, start_pos_to_token_idx)

    # Now distinguish the O cases from ones where we miss the tokenization
    entity_positions = _get_entity_positions(entities)
    _handle_not_an_entity(bilou, tokens, entity_positions, missing)

    return bilou


def _handle_entities(
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
                bilou[start_token_idx] = "U-%s" % label
            else:
                bilou[start_token_idx] = "B-%s" % label
                for i in range(start_token_idx + 1, end_token_idx):
                    bilou[i] = "I-%s" % label
                bilou[end_token_idx] = "L-%s" % label


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
