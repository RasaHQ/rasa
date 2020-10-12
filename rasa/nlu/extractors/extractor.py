from typing import Any, Dict, List, Text, Tuple, Optional

import rasa.shared.utils.io
from rasa.shared.constants import DOCS_URL_TRAINING_DATA_NLU
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.message import Message
from rasa.nlu.tokenizers.tokenizer import Token
from rasa.nlu.components import Component
from rasa.nlu.constants import (
    TOKENS_NAMES,
    ENTITY_ATTRIBUTE_CONFIDENCE_TYPE,
    ENTITY_ATTRIBUTE_CONFIDENCE_ROLE,
    ENTITY_ATTRIBUTE_CONFIDENCE_GROUP,
)
from rasa.shared.nlu.constants import (
    TEXT,
    INTENT,
    ENTITIES,
    ENTITY_ATTRIBUTE_VALUE,
    ENTITY_ATTRIBUTE_START,
    ENTITY_ATTRIBUTE_END,
    EXTRACTOR,
    ENTITY_ATTRIBUTE_TYPE,
    ENTITY_ATTRIBUTE_GROUP,
    ENTITY_ATTRIBUTE_ROLE,
    NO_ENTITY_TAG,
)


class EntityExtractor(Component):
    def add_extractor_name(
        self, entities: List[Dict[Text, Any]]
    ) -> List[Dict[Text, Any]]:
        for entity in entities:
            entity[EXTRACTOR] = self.name
        return entities

    def add_processor_name(self, entity: Dict[Text, Any]) -> Dict[Text, Any]:
        if "processors" in entity:
            entity["processors"].append(self.name)
        else:
            entity["processors"] = [self.name]

        return entity

    @staticmethod
    def filter_irrelevant_entities(extracted: list, requested_dimensions: set) -> list:
        """Only return dimensions the user configured"""

        if requested_dimensions:
            return [
                entity
                for entity in extracted
                if entity[ENTITY_ATTRIBUTE_TYPE] in requested_dimensions
            ]
        return extracted

    @staticmethod
    def find_entity(
        entity: Dict[Text, Any], text: Text, tokens: List[Token]
    ) -> Tuple[int, int]:
        offsets = [token.start for token in tokens]
        ends = [token.end for token in tokens]

        if entity[ENTITY_ATTRIBUTE_START] not in offsets:
            message = (
                "Invalid entity {} in example '{}': "
                "entities must span whole tokens. "
                "Wrong entity start.".format(entity, text)
            )
            raise ValueError(message)

        if entity[ENTITY_ATTRIBUTE_END] not in ends:
            message = (
                "Invalid entity {} in example '{}': "
                "entities must span whole tokens. "
                "Wrong entity end.".format(entity, text)
            )
            raise ValueError(message)

        start = offsets.index(entity[ENTITY_ATTRIBUTE_START])
        end = ends.index(entity[ENTITY_ATTRIBUTE_END]) + 1
        return start, end

    def filter_trainable_entities(
        self, entity_examples: List[Message]
    ) -> List[Message]:
        """Filters out untrainable entity annotations.

        Creates a copy of entity_examples in which entities that have
        `extractor` set to something other than
        self.name (e.g. 'CRFEntityExtractor') are removed.
        """

        filtered = []
        for message in entity_examples:
            entities = []
            for ent in message.get(ENTITIES, []):
                extractor = ent.get(EXTRACTOR)
                if not extractor or extractor == self.name:
                    entities.append(ent)
            data = message.data.copy()
            data[ENTITIES] = entities
            filtered.append(
                Message(
                    text=message.get(TEXT),
                    data=data,
                    output_properties=message.output_properties,
                    time=message.time,
                    features=message.features,
                )
            )

        return filtered

    def convert_predictions_into_entities(
        self,
        text: Text,
        tokens: List[Token],
        tags: Dict[Text, List[Text]],
        confidences: Optional[Dict[Text, List[float]]] = None,
    ) -> List[Dict[Text, Any]]:
        """
        Convert predictions into entities.

        Args:
            text: The text message.
            tokens: Message tokens without CLS token.
            tags: Predicted tags.
            confidences: Confidences of predicted tags.

        Returns:
            Entities.
        """
        import rasa.nlu.utils.bilou_utils as bilou_utils

        entities = []

        last_entity_tag = NO_ENTITY_TAG
        last_role_tag = NO_ENTITY_TAG
        last_group_tag = NO_ENTITY_TAG
        last_token_end = -1

        for idx, token in enumerate(tokens):
            current_entity_tag = self.get_tag_for(tags, ENTITY_ATTRIBUTE_TYPE, idx)

            if current_entity_tag == NO_ENTITY_TAG:
                last_entity_tag = NO_ENTITY_TAG
                last_token_end = token.end
                continue

            current_group_tag = self.get_tag_for(tags, ENTITY_ATTRIBUTE_GROUP, idx)
            current_group_tag = bilou_utils.tag_without_prefix(current_group_tag)
            current_role_tag = self.get_tag_for(tags, ENTITY_ATTRIBUTE_ROLE, idx)
            current_role_tag = bilou_utils.tag_without_prefix(current_role_tag)

            group_or_role_changed = (
                last_group_tag != current_group_tag or last_role_tag != current_role_tag
            )

            if bilou_utils.bilou_prefix_from_tag(current_entity_tag):
                # checks for new bilou tag
                # new bilou tag begins are not with I- , L- tags
                new_bilou_tag_starts = last_entity_tag != current_entity_tag and (
                    bilou_utils.LAST
                    != bilou_utils.bilou_prefix_from_tag(current_entity_tag)
                    and bilou_utils.INSIDE
                    != bilou_utils.bilou_prefix_from_tag(current_entity_tag)
                )

                # to handle bilou tags such as only I-, L- tags without B-tag
                # and handle multiple U-tags consecutively
                new_unigram_bilou_tag_starts = (
                    last_entity_tag == NO_ENTITY_TAG
                    or bilou_utils.UNIT
                    == bilou_utils.bilou_prefix_from_tag(current_entity_tag)
                )

                new_tag_found = (
                    new_bilou_tag_starts
                    or new_unigram_bilou_tag_starts
                    or group_or_role_changed
                )
                last_entity_tag = current_entity_tag
                current_entity_tag = bilou_utils.tag_without_prefix(current_entity_tag)
            else:
                new_tag_found = (
                    last_entity_tag != current_entity_tag or group_or_role_changed
                )
                last_entity_tag = current_entity_tag

            if new_tag_found:
                # new entity found
                entity = self._create_new_entity(
                    list(tags.keys()),
                    current_entity_tag,
                    current_group_tag,
                    current_role_tag,
                    token,
                    idx,
                    confidences,
                )
                entities.append(entity)
            elif token.start - last_token_end <= 1:
                # current token has the same entity tag as the token before and
                # the two tokens are only separated by at most one symbol (e.g. space,
                # dash, etc.)
                entities[-1][ENTITY_ATTRIBUTE_END] = token.end
                if confidences is not None:
                    self._update_confidence_values(entities, confidences, idx)
            else:
                # the token has the same entity tag as the token before but the two
                # tokens are separated by at least 2 symbols (e.g. multiple spaces,
                # a comma and a space, etc.)
                entity = self._create_new_entity(
                    list(tags.keys()),
                    current_entity_tag,
                    current_group_tag,
                    current_role_tag,
                    token,
                    idx,
                    confidences,
                )
                entities.append(entity)

            last_group_tag = current_group_tag
            last_role_tag = current_role_tag
            last_token_end = token.end

        for entity in entities:
            entity[ENTITY_ATTRIBUTE_VALUE] = text[
                entity[ENTITY_ATTRIBUTE_START] : entity[ENTITY_ATTRIBUTE_END]
            ]

        return entities

    @staticmethod
    def _update_confidence_values(
        entities: List[Dict[Text, Any]], confidences: Dict[Text, List[float]], idx: int
    ):
        # use the lower confidence value
        entities[-1][ENTITY_ATTRIBUTE_CONFIDENCE_TYPE] = min(
            entities[-1][ENTITY_ATTRIBUTE_CONFIDENCE_TYPE],
            confidences[ENTITY_ATTRIBUTE_TYPE][idx],
        )
        if ENTITY_ATTRIBUTE_ROLE in entities[-1]:
            entities[-1][ENTITY_ATTRIBUTE_CONFIDENCE_ROLE] = min(
                entities[-1][ENTITY_ATTRIBUTE_CONFIDENCE_ROLE],
                confidences[ENTITY_ATTRIBUTE_ROLE][idx],
            )
        if ENTITY_ATTRIBUTE_GROUP in entities[-1]:
            entities[-1][ENTITY_ATTRIBUTE_CONFIDENCE_GROUP] = min(
                entities[-1][ENTITY_ATTRIBUTE_CONFIDENCE_GROUP],
                confidences[ENTITY_ATTRIBUTE_GROUP][idx],
            )

    @staticmethod
    def get_tag_for(tags: Dict[Text, List[Text]], tag_name: Text, idx: int) -> Text:
        """Get the value of the given tag name from the list of tags.

        Args:
            tags: Mapping of tag name to list of tags;
            tag_name: The tag name of interest.
            idx: The index position of the tag.

        Returns:
            The tag value.
        """
        if tag_name in tags:
            return tags[tag_name][idx]
        return NO_ENTITY_TAG

    @staticmethod
    def _create_new_entity(
        tag_names: List[Text],
        entity_tag: Text,
        group_tag: Text,
        role_tag: Text,
        token: Token,
        idx: int,
        confidences: Optional[Dict[Text, List[float]]] = None,
    ) -> Dict[Text, Any]:
        """Create a new entity.

        Args:
            tag_names: The tag names to include in the entity.
            entity_tag: The entity type value.
            group_tag: The entity group value.
            role_tag: The entity role value.
            token: The token.
            confidence: The confidence value.

        Returns:
            Created entity.
        """
        entity = {
            ENTITY_ATTRIBUTE_TYPE: entity_tag,
            ENTITY_ATTRIBUTE_START: token.start,
            ENTITY_ATTRIBUTE_END: token.end,
        }

        if confidences is not None:
            entity[ENTITY_ATTRIBUTE_CONFIDENCE_TYPE] = confidences[
                ENTITY_ATTRIBUTE_TYPE
            ][idx]

        if ENTITY_ATTRIBUTE_ROLE in tag_names and role_tag != NO_ENTITY_TAG:
            entity[ENTITY_ATTRIBUTE_ROLE] = role_tag
            if confidences is not None:
                entity[ENTITY_ATTRIBUTE_CONFIDENCE_ROLE] = confidences[
                    ENTITY_ATTRIBUTE_ROLE
                ][idx]
        if ENTITY_ATTRIBUTE_GROUP in tag_names and group_tag != NO_ENTITY_TAG:
            entity[ENTITY_ATTRIBUTE_GROUP] = group_tag
            if confidences is not None:
                entity[ENTITY_ATTRIBUTE_CONFIDENCE_GROUP] = confidences[
                    ENTITY_ATTRIBUTE_GROUP
                ][idx]

        return entity

    @staticmethod
    def check_correct_entity_annotations(training_data: TrainingData) -> None:
        """Check if entities are correctly annotated in the training data.

        If the start and end values of an entity do not match any start and end values
        of the respected token, we define an entity as misaligned and log a warning.

        Args:
            training_data: The training data.
        """
        for example in training_data.entity_examples:
            entity_boundaries = [
                (entity[ENTITY_ATTRIBUTE_START], entity[ENTITY_ATTRIBUTE_END])
                for entity in example.get(ENTITIES)
            ]
            token_start_positions = [t.start for t in example.get(TOKENS_NAMES[TEXT])]
            token_end_positions = [t.end for t in example.get(TOKENS_NAMES[TEXT])]

            for entity_start, entity_end in entity_boundaries:
                if (
                    entity_start not in token_start_positions
                    or entity_end not in token_end_positions
                ):
                    rasa.shared.utils.io.raise_warning(
                        f"Misaligned entity annotation in message '{example.get(TEXT)}' "
                        f"with intent '{example.get(INTENT)}'. Make sure the start and "
                        f"end values of entities in the training data match the token "
                        f"boundaries (e.g. entities don't include trailing whitespaces "
                        f"or punctuation).",
                        docs=DOCS_URL_TRAINING_DATA_NLU,
                    )
                    break
