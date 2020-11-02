import logging
from typing import Any, Dict, Text

from rasa.shared.nlu.constants import INTENT, ENTITIES, TEXT
from rasa.shared.nlu.training_data.formats.readerwriter import JsonTrainingDataReader

from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.message import Message

logger = logging.getLogger(__name__)


class WitReader(JsonTrainingDataReader):
    def read_from_json(self, js: Dict[Text, Any], **kwargs: Any):
        """Loads training data stored in the WIT.ai data format."""
        training_examples = []

        for s in js["data"]:
            entities = s.get(ENTITIES)
            if entities is None:
                continue
            text = s.get(TEXT)
            intents = [e["value"] for e in entities if e["entity"] == INTENT]
            intent = intents[0].strip('"') if intents else None

            entities = [
                e
                for e in entities
                if ("start" in e and "end" in e and e["entity"] != INTENT)
            ]
            for e in entities:
                # for some reason wit adds additional quotes around entities
                e["value"] = e["value"].strip('"')

            data = {}
            if intent:
                data[INTENT] = intent
            if entities is not None:
                data[ENTITIES] = entities
            data[TEXT] = text
            training_examples.append(Message(data=data))
        return TrainingData(training_examples)
