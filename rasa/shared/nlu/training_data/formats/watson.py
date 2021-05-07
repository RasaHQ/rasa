import logging
from typing import Any, Dict, List, Text

from rasa.shared.nlu.constants import (
    INTENT,
    ENTITIES,
    TEXT,
    VALID_FEATURE_TYPES,
)
from rasa.shared.nlu.training_data.formats.readerwriter import JsonTrainingDataReader
from rasa.shared.nlu.training_data.util import transform_entity_synonyms
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.message import Message

logger = logging.getLogger(__name__)


class WatsonReader(JsonTrainingDataReader):
    """Reads Watson training data and train a Rasa NLU model."""

    def read_from_json(self, js: Dict[Text, Any], **kwargs: Any) -> TrainingData:
        """Loads training data stored in the IBM Watson data format."""
        training_examples = []
        entity_synonyms = self._entity_synonyms(js)
        entity_synonyms = transform_entity_synonyms(entity_synonyms)
        all_entities = self._list_all_entities(js)
        for intent in js.get("intents"):
            examples = intent.get("examples")
            intent = intent.get("intent")
            if not examples:
                continue
            for text in examples:
                utterance = text.get("text")
                example_with_entities = self._unpack_entity_examples(
                    js=js,
                    text=utterance,
                    intent=intent,
                    training_examples=training_examples,
                    all_entities=all_entities,
                )
                if utterance in example_with_entities:
                    continue

                self._add_training_examples_without_entities(
                    training_examples, intent, utterance
                )

        return TrainingData(training_examples, entity_synonyms)

    @staticmethod
    def _add_training_examples_without_entities(
        training_examples: List, intent: str, text: str
    ) -> None:
        training_examples.append(Message(data={INTENT: intent, TEXT: text}))

    def _unpack_entity_examples(
        self,
        js: Dict[Text, Any],
        text: str,
        intent: str,
        training_examples,
        all_entities,
    ) -> List:
        examples_with_entities = []
        all_entity_names = set().union(*(d.keys() for d in all_entities))
        for entity_name in all_entity_names:
            if entity_name in text:
                examples_with_entities.append(text)
                values_of_entity = [
                    a_dict[entity_name]
                    for a_dict in all_entities
                    if entity_name in a_dict
                ]
                if values_of_entity is None:
                    continue
                for val in values_of_entity[0]:
                    entities = []
                    unpack_text = text.replace(entity_name, val.get("value")).replace(
                        "@", ""
                    )
                    start_index = unpack_text.index(val.get("value"))
                    end_index = start_index + len(val.get("value"))
                    entities.append(
                        {
                            "entity": entity_name,
                            "start": start_index,
                            "end": end_index,
                            "value": val.get("value"),
                        }
                    )
                    training_examples.append(
                        Message(
                            data={INTENT: intent, TEXT: unpack_text, ENTITIES: entities}
                        )
                    )

        return examples_with_entities

    @staticmethod
    def _list_all_entities(js: Dict[Text, Any]) -> List:
        all_entities = []
        entities = js.get("entities")
        if not entities:
            for entity in entities:
                all_entities.append({entity.get("entity"): entity.get("values")})
        return all_entities

    def _entity_synonyms(self, js: Dict[Text, Any]) -> List:
        entity_synonyms = []
        entities = js.get("entities")
        if not entities:
            for entity in entities:
                for val in entity.get("values"):
                    entity_synonyms.append(
                        {"value": val.get("value"), "synonyms": val.get("synonyms"),}
                    )
        return entity_synonyms
