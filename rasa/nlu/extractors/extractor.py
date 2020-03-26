from typing import Any, Dict, List, Text, Tuple, Optional

from nlu.tokenizers.tokenizer import Token
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
        Checks if multiple entity labels are assigned to one word or if an entity label
        is assigned to just a part of a word.

        This might happen if you are using a tokenizer that splits up words into
        sub-words and different entity labels are assigned to the individual sub-words.
        If multiple entity labels are assigned to the word, we keep the entity label
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

        Returns: updated list of entities
        """
        word_clusters = self._word_clusters(message.get(TOKENS_NAMES[TEXT]), entities)

        entity_indices_to_remove = set()

        for cluster in word_clusters:
            entity_indices = cluster["entity_indices"]

            if not keep:
                entity_indices_to_remove.update(entity_indices)
                continue

            idx = self._entity_index_to_keep(entities, entity_indices)

            if idx is None:
                entity_indices_to_remove.update(entity_indices)
            else:
                # just keep one entity
                entity_indices.remove(idx)
                entity_indices_to_remove.update(entity_indices)

                # update that entity to cover the complete word
                entities[idx]["start"] = cluster["start"]
                entities[idx]["end"] = cluster["end"]
                entities[idx]["value"] = cluster["text"]

        # sort indices to remove entries at the end of the list first
        # to avoid index out of range errors
        for idx in sorted(entity_indices_to_remove, reverse=True):
            entities.remove(entities[idx])

        return entities

    def _word_clusters(
        self, tokens: List[Token], entities: List[Dict[Text, Any]]
    ) -> List[Dict[Text, Any]]:
        """
        Build cluster of tokens and entities that belong to one word.

        Args:
            tokens: list of tokens
            entities: list of detected entities by the entity extractor

        Returns:
            a list of clusters containing start and end position, text, and entity
            indices
        """

        # get all token indices that belong to one word
        token_clusters = self._token_clusters(tokens)

        if not token_clusters:
            return []

        word_clusters = []
        for token_cluster in token_clusters:
            # get start and end position and text of complete word
            # needed to update the final entity later
            start_position = token_cluster[0].start
            end_position = token_cluster[-1].end
            text = "".join(t.text for t in token_cluster)

            entity_indices = [
                idx
                for idx, e in enumerate(entities)
                if e["start"] >= start_position and e["end"] <= end_position
            ]

            # we are just interested in words split up into multiple tokens that
            # got an entity assigned
            if entity_indices:
                word_clusters.append(
                    {
                        "start": start_position,
                        "end": end_position,
                        "text": text,
                        "entity_indices": entity_indices,
                    }
                )

        return word_clusters

    @staticmethod
    def _token_clusters(tokens: List[Token]) -> List[List[Token]]:
        """
        Get tokens that belong to one word.

        Args:
            tokens: list of tokens

        Returns: list of token clusters

        """
        token_index_clusters = []

        for idx in range(1, len(tokens)):
            # two token belong to the same word if there is no other character
            # between them
            if tokens[idx].start == tokens[idx - 1].end:
                if token_index_clusters and token_index_clusters[-1][-1] == idx - 1:
                    token_index_clusters[-1].append(idx)
                else:
                    token_index_clusters.append([idx - 1, idx])

        token_clusters = []

        for cluster in token_index_clusters:
            cluster.sort()
            token_clusters.append([tokens[idx] for idx in cluster])

        return token_clusters

    @staticmethod
    def _entity_index_to_keep(
        entities: List[Dict[Text, Any]], entity_indices: List[int]
    ) -> Optional[int]:
        """
        Determine the entity index to keep. If we just have one entity index, i.e.
        candidate, we return the index of that candidate. If we have multiple
        candidate, we return the index of the entity value with the highest
        confidence score. If no confidence score is present, no entity label will
        be kept.

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
