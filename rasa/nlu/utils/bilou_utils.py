from typing import List, Tuple, Text, Optional, Dict

from rasa.nlu.tokenizers.tokenizer import Token
from rasa.nlu.training_data import Message
from rasa.nlu.training_data import TrainingData
from rasa.nlu.constants import (
    ENTITIES_ATTRIBUTE,
    TOKENS_NAMES,
    TEXT_ATTRIBUTE,
    BILOU_ENTITIES_ATTRIBUTE,
)

BILOU_PREFIXES = ["B-", "I-", "U-", "L-"]


def entity_name_from_tag(tag: Text) -> Text:
    """Remove the BILOU prefix from the given tag."""
    if tag[:2] in BILOU_PREFIXES:
        return tag[2:]
    return tag


def bilou_from_tag(tag: Text) -> Optional[Text]:
    """Get the BILOU prefix (without -) from the given tag."""
    if len(tag) >= 2 and tag[1] == "-" and tag[:2] in BILOU_PREFIXES:
        return tag[0].upper()
    return None


def tags_to_ids(message: Message, tag_id_dict: Dict[Text, int]) -> List[int]:
    """Maps the entity tags of the message to the ids of the provided dict."""
    if message.get(BILOU_ENTITIES_ATTRIBUTE):
        _tags = [
            tag_id_dict[_tag] if _tag in tag_id_dict else tag_id_dict["O"]
            for _tag in message.get(BILOU_ENTITIES_ATTRIBUTE)
        ]
    else:
        _tags = [tag_id_dict["O"] for _ in message.get(TOKENS_NAMES[TEXT_ATTRIBUTE])]

    return _tags


def remove_bilou_prefixes(tags: List[Text]) -> List[Text]:
    """Remove the BILOU prefixes from the given tags."""
    return [entity_name_from_tag(t) for t in tags]


def build_tag_id_dict(training_data: TrainingData) -> Dict[Text, int]:
    """Create a mapping of unique tags to ids."""
    distinct_tag_ids = set(
        [
            entity_name_from_tag(e)
            for example in training_data.training_examples
            if example.get(BILOU_ENTITIES_ATTRIBUTE)
            for e in example.get(BILOU_ENTITIES_ATTRIBUTE)
        ]
    ) - {"O"}

    tag_id_dict = {
        f"{prefix}{tag_id}": idx_1 * len(BILOU_PREFIXES) + idx_2 + 1
        for idx_1, tag_id in enumerate(sorted(distinct_tag_ids))
        for idx_2, prefix in enumerate(BILOU_PREFIXES)
    }
    tag_id_dict["O"] = 0

    return tag_id_dict


def apply_bilou_schema(training_data: TrainingData):
    """Obtains a list of BILOU entity tags and sets them on the corresponding
    message."""
    for message in training_data.training_examples:
        entities = message.get(ENTITIES_ATTRIBUTE)

        if not entities:
            continue

        entities = map_message_entities(message)
        output = bilou_tags_from_offsets(
            message.get(TOKENS_NAMES[TEXT_ATTRIBUTE]), entities
        )

        message.set(BILOU_ENTITIES_ATTRIBUTE, output)


def map_message_entities(message: Message) -> List[Tuple[int, int, Text]]:
    """Maps the entities of the given message to their start, end, and tag values."""

    def convert_entity(entity):
        return entity["start"], entity["end"], entity["entity"]

    return [convert_entity(entity) for entity in message.get(ENTITIES_ATTRIBUTE, [])]


def bilou_tags_from_offsets(
    tokens: List[Token], entities: List[Tuple[int, int, Text]], missing: Text = "O"
) -> List[Text]:
    """Creates a list of BILOU tags for the given list of tokens and entities."""

    # From spacy.spacy.GoldParse, under MIT License
    starts = {token.start: i for i, token in enumerate(tokens)}
    ends = {token.end: i for i, token in enumerate(tokens)}
    bilou = ["-" for _ in tokens]

    # Handle entity cases
    for start_char, end_char, label in entities:
        start_token = starts.get(start_char)
        end_token = ends.get(end_char)

        # Only interested if the tokenization is correct
        if start_token is not None and end_token is not None:
            if start_token == end_token:
                bilou[start_token] = "U-%s" % label
            else:
                bilou[start_token] = "B-%s" % label
                for i in range(start_token + 1, end_token):
                    bilou[i] = "I-%s" % label
                bilou[end_token] = "L-%s" % label

    # Now distinguish the O cases from ones where we miss the tokenization
    entity_chars = set()
    for start_char, end_char, label in entities:
        for i in range(start_char, end_char):
            entity_chars.add(i)

    for n, token in enumerate(tokens):
        for i in range(token.start, token.end):
            if i in entity_chars:
                break
        else:
            bilou[n] = missing

    return bilou
