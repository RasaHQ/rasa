from typing import Any, Dict, List, Text, Tuple, Optional, Union

from rasa.nlu.tokenizers.tokenizer import Token
from rasa.nlu.components import Component
from rasa.nlu.constants import EXTRACTOR, ENTITIES, TOKENS_NAMES, TEXT
from rasa.nlu.training_data import Message


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

            if idx is None:
                entity_indices_to_remove.update(entity_indices)
            else:
                # keep just one entity
                entity_indices.remove(idx)
                entity_indices_to_remove.update(entity_indices)

                # update that entity to cover the complete word(s)
                entities[idx]["start"] = misaligned_entity["start"]
                entities[idx]["end"] = misaligned_entity["end"]
                entities[idx]["value"] = message.text[
                    misaligned_entity["start"] : misaligned_entity["end"]
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
                        "start": start_position,
                        "end": end_position,
                        "entity_indices": [entity_idx],
                    }
                )
            else:
                misaligned_entities[_idx]["entity_indices"].append(entity_idx)

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
            if cluster["start"] == start_position and cluster["end"] == end_position:
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
                token_cluster[0].start <= entity["start"] <= token_cluster[-1].end
            )
            entity_ends_inside_cluster = (
                token_cluster[0].start <= entity["end"] <= token_cluster[-1].end
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
            entities[idx]["confidence"]
            for idx in entity_indices
            if "confidence" in entities[idx]
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
                if entity["entity"] in requested_dimensions
            ]
        else:
            return extracted

    @staticmethod
    def find_entity(ent, text, tokens) -> Tuple[int, int]:
        offsets = [token.start for token in tokens]
        ends = [token.end for token in tokens]

        if ent["start"] not in offsets:
            message = (
                "Invalid entity {} in example '{}': "
                "entities must span whole tokens. "
                "Wrong entity start.".format(ent, text)
            )
            raise ValueError(message)

        if ent["end"] not in ends:
            message = (
                "Invalid entity {} in example '{}': "
                "entities must span whole tokens. "
                "Wrong entity end.".format(ent, text)
            )
            raise ValueError(message)

        start = offsets.index(ent["start"])
        end = ends.index(ent["end"]) + 1
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
