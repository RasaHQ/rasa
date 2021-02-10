import logging
from typing import Any, Dict, List, Text
import re

from rasa.shared.nlu.constants import (
    INTENT,
    ENTITIES,
    TEXT,
    ENTITY_ATTRIBUTE_ROLE,
    ENTITY_ATTRIBUTE_TYPE,
)
from rasa.shared.nlu.training_data.formats.readerwriter import JsonTrainingDataReader
from rasa.shared.nlu.training_data.util import transform_entity_synonyms
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.message import Message

logger = logging.getLogger(__name__)


class WatsonReader(JsonTrainingDataReader):
    def read_from_json(self, js: Dict[Text, Any], **kwargs: Any):
        """Loads training data stored in the IBM Watson data format."""
        training_examples = []
        unpacked_texts = []
        entity_synonyms = self._entity_synonyms(js)
        entity_synonyms = transform_entity_synonyms(entity_synonyms)
        for s in js["intents"]:
            examples = s["examples"]
            if examples == []:
                continue
            for text in examples:
                example_already_added = self._unpack_entity_examples(
                    js=js,
                    text=text,
                    intent=s["intent"],
                    training_examples=training_examples,
                )
                if text["text"] in example_already_added:
                    continue
                else:
                    self._add_training_examples(
                        training_examples, s["intent"], text["text"]
                    )

        return TrainingData(training_examples, entity_synonyms)

    def _add_training_examples(self, training_examples: List, intent: str, text: str):
        data = {}
        if intent:
            data[INTENT] = intent
        data[TEXT] = text
        training_examples.append(Message(data=data))

    def _unpack_entity_examples(
        self, js: Dict[Text, Any], text: Dict, intent: str, training_examples
    ):
        all_entities = self._list_all_entities(js)
        example_already_added = []
        for entity in all_entities:
            for key, value in entity.items():
                if key in text["text"]:
                    example_already_added.append(text["text"])
                    for val in value:
                        entities = []
                        unpack_text = (
                            text["text"].replace(key, val["value"]).replace("@", "")
                        )
                        start_index = unpack_text.index(val["value"])
                        end_index = start_index + len(val["value"])
                        entities.append(
                            {
                                "entity": key,
                                "start": start_index,
                                "end": end_index,
                                "value": val["value"],
                            }
                        )
                        data = {}
                        data[INTENT] = intent
                        data[TEXT] = unpack_text
                        data[ENTITIES] = entities
                        training_examples.append(Message(data=data))
        return example_already_added

    @staticmethod
    def _list_all_entities(js: Dict[Text, Any]):
        entities = []
        for entity in js["entities"]:
            entities.append({entity["entity"]: entity["values"]})
        return entities

    def _entity_synonyms(self, js: Dict[Text, Any]):
        entity_synonyms = []

        for entity in js["entities"]:
            for val in entity["values"]:
                entity_synonyms.append(
                    {"value": val["value"], "synonyms": val["synonyms"]}
                )
        return entity_synonyms
