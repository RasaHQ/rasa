from typing import Any, Dict, List, Text, Tuple, Optional, Union

from rasa.constants import DOCS_URL_TRAINING_DATA_NLU
from rasa.nlu.training_data import TrainingData
from rasa.nlu.tokenizers.tokenizer import Token
from rasa.nlu.components import Component
from rasa.nlu.constants import (
    EXTRACTOR,
    ENTITIES,
    TOKENS_NAMES,
    TEXT,
    NO_ENTITY_TAG,
    ENTITY_ATTRIBUTE_TYPE,
    ENTITY_ATTRIBUTE_GROUP,
    ENTITY_ATTRIBUTE_ROLE,
    ENTITY_ATTRIBUTE_VALUE,
    ENTITY_ATTRIBUTE_CONFIDENCE_TYPE,
    ENTITY_ATTRIBUTE_CONFIDENCE_ROLE,
    ENTITY_ATTRIBUTE_CONFIDENCE_GROUP,
    ENTITY_ATTRIBUTE_START,
    ENTITY_ATTRIBUTE_END,
    INTENT,
)
from rasa.nlu.training_data import Message
import rasa.utils.common as common_utils


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

    def clean_up_entities(
        self, message: Message, entities: List[Dict[Text, Any]], keep: bool = True
    ) -> List[Dict[Text, Any]]:
        """
        Check if multiple entity labels are assigned to one word or if an entity label
        is assigned to just a part of a word or if an entity label covers multiple
        words, but one word just partly.

        This might happen if you are using a tokenizer that splits up words into
        sub-words and different entity labels are assigned to the individual sub-words.
        If multiple entity labels are assigned to one word, we keep the entity label
        with the highest confidence as entity label for that word. If just a part
        of the word is annotated, that entity label is taken for the complete word.
        If you set 'keep' to 'False', all entity labels for the word will be removed.

        Args:
            message: message object
            entities: list of entities
            keep:
                If set to 'True', the entity label with the highest confidence is kept
                if multiple entity labels are assigned to one word. If set to 'False'
                all entity labels for that word will be removed.

        Returns:
            Updated entities.
        """
        misaligned_entities = self._get_misaligned_entities(
            message.get(TOKENS_NAMES[TEXT]), entities
        )

        entity_indices_to_remove = set()

        for misaligned_entity in misaligned_entities:
            # entity indices involved in the misalignment
            entity_indices = misaligned_entity["entity_indices"]

            if not keep:
                entity_indices_to_remove.update(entity_indices)
                continue

            idx = self._entity_index_to_keep(entities, entity_indices)

            if idx is None or idx not in entity_indices:
                entity_indices_to_remove.update(entity_indices)
            else:
                # keep just one entity
                entity_indices.remove(idx)
                entity_indices_to_remove.update(entity_indices)

                # update that entity to cover the complete word(s)
                entities[idx][ENTITY_ATTRIBUTE_START] = misaligned_entity[
                    ENTITY_ATTRIBUTE_START
                ]
                entities[idx][ENTITY_ATTRIBUTE_END] = misaligned_entity[
                    ENTITY_ATTRIBUTE_END
                ]
                entities[idx][ENTITY_ATTRIBUTE_VALUE] = message.text[
                    misaligned_entity[ENTITY_ATTRIBUTE_START] : misaligned_entity[
                        ENTITY_ATTRIBUTE_END
                    ]
                ]

        # sort indices to remove entries at the end of the list first
        # to avoid index out of range errors
        for idx in sorted(entity_indices_to_remove, reverse=True):
            entities.remove(entities[idx])

        return entities

    def _get_misaligned_entities(
        self, tokens: List[Token], entities: List[Dict[Text, Any]]
    ) -> List[Dict[Text, Any]]:
        """Identify entities and tokens that are misaligned.

        Misaligned entities are those that apply only to a part of a word, i.e.
        sub-word.

        Args:
            tokens: list of tokens
            entities: list of detected entities by the entity extractor

        Returns:
            Misaligned entities including the start and end position
            of the final entity in the text and entity indices that are part of this
            misalignment.
        """
        if not tokens:
            return []

        # group tokens: one token cluster corresponds to one word
        token_clusters = self._token_clusters(tokens)

        # added for tests, should only happen if tokens are not set or len(tokens) == 1
        if not token_clusters:
            return []

        misaligned_entities = []
        for entity_idx, entity in enumerate(entities):
            # get all tokens that are covered/touched by the entity
            entity_tokens = self._tokens_of_entity(entity, token_clusters)

            if len(entity_tokens) == 1:
                # entity covers exactly one word
                continue

            # get start and end position of complete word
            # needed to update the final entity later
            start_position = entity_tokens[0].start
            end_position = entity_tokens[-1].end

            # check if an entity was already found that covers the exact same word(s)
            _idx = self._misaligned_entity_index(
                misaligned_entities, start_position, end_position
            )

            if _idx is None:
                misaligned_entities.append(
                    {
                        ENTITY_ATTRIBUTE_START: start_position,
                        ENTITY_ATTRIBUTE_END: end_position,
                        "entity_indices": [entity_idx],
                    }
                )
            else:
                # pytype: disable=attribute-error
                misaligned_entities[_idx]["entity_indices"].append(entity_idx)
                # pytype: enable=attribute-error

        return misaligned_entities

    @staticmethod
    def _misaligned_entity_index(
        word_entity_cluster: List[Dict[Text, Union[int, List[int]]]],
        start_position: int,
        end_position: int,
    ) -> Optional[int]:
        """Get index of matching misaligned entity.

        Args:
            word_entity_cluster: word entity cluster
            start_position: start position
            end_position: end position

        Returns:
            Index of the misaligned entity that matches the provided start and end
            position.
        """
        for idx, cluster in enumerate(word_entity_cluster):
            if (
                cluster[ENTITY_ATTRIBUTE_START] == start_position
                and cluster[ENTITY_ATTRIBUTE_END] == end_position
            ):
                return idx
        return None

    @staticmethod
    def _tokens_of_entity(
        entity: Dict[Text, Any], token_clusters: List[List[Token]]
    ) -> List[Token]:
        """Get all tokens of token clusters that are covered by the entity.

        The entity can cover them completely or just partly.

        Args:
            entity: the entity
            token_clusters: list of token clusters

        Returns:
            Token clusters that belong to the provided entity.

        """
        entity_tokens = []
        for token_cluster in token_clusters:
            entity_starts_inside_cluster = (
                token_cluster[0].start
                <= entity[ENTITY_ATTRIBUTE_START]
                <= token_cluster[-1].end
            )
            entity_ends_inside_cluster = (
                token_cluster[0].start
                <= entity[ENTITY_ATTRIBUTE_END]
                <= token_cluster[-1].end
            )

            if entity_starts_inside_cluster or entity_ends_inside_cluster:
                entity_tokens += token_cluster
        return entity_tokens

    @staticmethod
    def _token_clusters(tokens: List[Token]) -> List[List[Token]]:
        """Build clusters of tokens that belong to one word.

        Args:
            tokens: list of tokens

        Returns:
            Token clusters.

        """
        # token cluster = list of token indices that belong to one word
        token_index_clusters = []

        # start at 1 in order to check if current token and previous token belong
        # to the same word
        for token_idx in range(1, len(tokens)):
            previous_token_idx = token_idx - 1
            # two tokens belong to the same word if there is no other character
            # between them
            if tokens[token_idx].start == tokens[previous_token_idx].end:
                # a word was split into multiple tokens
                token_cluster_already_exists = (
                    token_index_clusters
                    and token_index_clusters[-1][-1] == previous_token_idx
                )
                if token_cluster_already_exists:
                    token_index_clusters[-1].append(token_idx)
                else:
                    token_index_clusters.append([previous_token_idx, token_idx])
            else:
                # the token corresponds to a single word
                if token_idx == 1:
                    token_index_clusters.append([previous_token_idx])
                token_index_clusters.append([token_idx])

        return [[tokens[idx] for idx in cluster] for cluster in token_index_clusters]

    @staticmethod
    def _entity_index_to_keep(
        entities: List[Dict[Text, Any]], entity_indices: List[int]
    ) -> Optional[int]:
        """
        Determine the entity index to keep.

        If we just have one entity index, i.e. candidate, we return the index of that
        candidate. If we have multiple candidates, we return the index of the entity
        value with the highest confidence score. If no confidence score is present,
        no entity label will be kept.

        Args:
            entities: the full list of entities
            entity_indices: the entity indices to consider

        Returns: the idx of the entity to keep
        """
        if len(entity_indices) == 1:
            return entity_indices[0]

        confidences = [
            entities[idx][ENTITY_ATTRIBUTE_CONFIDENCE_TYPE]
            for idx in entity_indices
            if ENTITY_ATTRIBUTE_CONFIDENCE_TYPE in entities[idx]
        ]

        # we don't have confidence values for all entity labels
        if len(confidences) != len(entity_indices):
            return None

        return confidences.index(max(confidences))

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
                    text=message.text,
                    data=data,
                    output_properties=message.output_properties,
                    time=message.time,
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
        entities = []

        last_entity_tag = NO_ENTITY_TAG
        last_role_tag = NO_ENTITY_TAG
        last_group_tag = NO_ENTITY_TAG

        for idx, token in enumerate(tokens):
            current_entity_tag = self.get_tag_for(tags, ENTITY_ATTRIBUTE_TYPE, idx)

            if current_entity_tag == NO_ENTITY_TAG:
                last_entity_tag = NO_ENTITY_TAG
                continue

            current_group_tag = self.get_tag_for(tags, ENTITY_ATTRIBUTE_GROUP, idx)
            current_role_tag = self.get_tag_for(tags, ENTITY_ATTRIBUTE_ROLE, idx)

            new_tag_found = (
                last_entity_tag != current_entity_tag
                or last_group_tag != current_group_tag
                or last_role_tag != current_role_tag
            )

            if new_tag_found:
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
            else:
                entities[-1][ENTITY_ATTRIBUTE_END] = token.end
                if confidences is not None:
                    self._update_confidence_values(entities, confidences, idx)

            last_entity_tag = current_entity_tag
            last_group_tag = current_group_tag
            last_role_tag = current_role_tag

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
                    common_utils.raise_warning(
                        f"Misaligned entity annotation in message '{example.text}' "
                        f"with intent '{example.get(INTENT)}'. Make sure the start and "
                        f"end values of entities in the training data match the token "
                        f"boundaries (e.g. entities don't include trailing whitespaces "
                        f"or punctuation).",
                        docs=DOCS_URL_TRAINING_DATA_NLU,
                    )
                    break
