from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

from rasa_nlu.training_data import Message, TrainingData
from rasa_nlu.training_data.formats.readerwriter import JsonTrainingDataReader

logger = logging.getLogger(__name__)


class WitReader(JsonTrainingDataReader):

    def read_from_json(self, js, **kwargs):
        # type: (Text, Any) -> TrainingData
        """Loads training data stored in the WIT.ai data format."""

        training_examples = []

        for s in js["data"]:
            entities = s.get("entities")
            if entities is None:
                continue
            text = s.get("text")
            intents = [e["value"] for e in entities if e["entity"] == 'intent']
            intent = intents[0].strip("\"") if intents else None

            entities = [e
                        for e in entities
                        if ("start" in e and "end" in e and
                            e["entity"] != 'intent')]
            for e in entities:
                # for some reason wit adds additional quotes around entity values
                e["value"] = e["value"].strip("\"")

            data = {}
            if intent:
                data["intent"] = intent
            if entities is not None:
                data["entities"] = entities
            training_examples.append(Message(text, data))
        return TrainingData(training_examples)
