from typing import Any, Dict, List, Text, Tuple, Optional

from rasa.nlu.components import Component
from rasa.nlu.constants import EXTRACTOR, ENTITIES
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
        self, entities: List[Dict[Text, Any]], keep: bool = True
    ) -> List[Dict[Text, Any]]:
        """
        Checks if multiple entity labels are assigned to one word.

        This might happen if you are using a tokenizer that splits up words into
        sub-words and different entity labels are assigned to the individual sub-words.
        In such a case keep the entity label with the highest confidence as entity
        label for that word. If you set 'keep' to 'False', all entity labels for
        that word will be removed.

        Args:
            entities: list of entities
            keep:
                If set to 'True', the entity label with the highest confidence is kept
                if multiple entity labels are assigned to one word. If set to 'False'
                all entity labels for that word will be removed.

        Returns: updated list of entities
        """
        if len(entities) <= 1:
            return entities

        entity_indices: List[List[int]] = []

        # get indices of entity labels that belong to one word
        for idx in range(1, len(entities)):
            if entities[idx]["start"] == entities[idx - 1]["end"]:
                if entity_indices and entity_indices[-1][-1] == idx - 1:
                    entity_indices[-1].append(idx)
                else:
                    entity_indices.append([idx - 1, idx])

        entities_to_remove = set()

        for indices in entity_indices:
            if not keep:
                entities_to_remove.update(indices)
                continue

            start = entities[indices[0]]["start"]
            end = entities[indices[-1]]["end"]
            value = "".join(entities[idx]["value"] for idx in indices)
            idx = self._get_highest_confidence_idx(entities, indices)

            if idx is None:
                entities_to_remove.update(indices)
            else:
                indices.remove(idx)
                entities_to_remove.update(indices)
                entities[idx]["start"] = start
                entities[idx]["end"] = end
                entities[idx]["value"] = value

        entities_to_remove = sorted(entities_to_remove, reverse=True)

        for idx in entities_to_remove:
            entities.remove(entities[idx])

        return entities

    @staticmethod
    def _get_highest_confidence_idx(
        entities: List[Dict[Text, Any]], indices: List[int]
    ) -> Optional[int]:
        """
        Args:
            entities: the full list of entities
            indices: the indices to consider

        Returns: the idx of the entity label with the highest confidence.
        """
        confidences = [
            entities[idx]["confidence"]
            for idx in indices
            if "confidence" in entities[idx]
        ]

        if len(confidences) != len(indices):
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
