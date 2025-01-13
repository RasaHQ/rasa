import logging
from typing import Any, Dict, Text

from rasa.shared.core.constants import USER_INTENT_OUT_OF_SCOPE
from rasa.shared.nlu.constants import (
    ENTITIES,
    ENTITY_ATTRIBUTE_ROLE,
    ENTITY_ATTRIBUTE_TYPE,
    ENTITY_ATTRIBUTE_VALUE,
    INTENT,
    TEXT,
)
from rasa.shared.nlu.training_data.formats.readerwriter import JsonTrainingDataReader
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData

logger = logging.getLogger(__name__)


class WitReader(JsonTrainingDataReader):
    def read_from_json(self, js: Dict[Text, Any], **kwargs: Any) -> TrainingData:
        """Loads training data stored in the WIT.ai data format."""
        training_examples = []

        for s in js["utterances"]:
            entities = s.get(ENTITIES)
            if entities is None:
                continue
            text = s.get(TEXT)
            # Out-of-scope WIT utterances won't have the intent field set,
            # and that's the reason why we set it to `USER_INTENT_OUT_OF_SCOPE` by
            # default.
            intent = s.get("intent", USER_INTENT_OUT_OF_SCOPE)

            for e in entities:
                entity_name = e["entity"]
                if ":" not in entity_name:
                    continue
                (name, role) = entity_name.rsplit(":", 1)
                e[ENTITY_ATTRIBUTE_TYPE] = name
                e[ENTITY_ATTRIBUTE_ROLE] = role
                e[ENTITY_ATTRIBUTE_VALUE] = e.pop("body", None)

            data = {}
            if intent:
                data[INTENT] = intent
            if entities is not None:
                data[ENTITIES] = entities
            data[TEXT] = text
            training_examples.append(Message(data=data))
        return TrainingData(training_examples)
