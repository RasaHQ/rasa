import os
import logging
from typing import Any, Dict, Optional, Text

from rasa.nlu import utils
from rasa.nlu.components import Component
from rasa.nlu.training_data import Message

logger = logging.getLogger(__name__)


class KeywordIntentClassifier(Component):
    """Intent classifier using simple keyword matching.


    The classifier takes a list of keywords and associated intents as an input.
    A input sentence is checked for the keywords and the intent is returned.

    """

    name = "intent_classifier_keyword"

    provides = ["intent"]

    intent_keyword_map = {}

    def train(self,
              training_data: 'TrainingData',
              cfg: Optional['RasaNLUModelConfig'] = None,
              **kwargs: Any) -> None:

        all_intents = list(training_data.intents)

        for intent in all_intents:
            self.intent_keyword_map[intent] = []
            for example in training_data.intent_examples:
                if example.data['intent'] == intent:
                    self.intent_keyword_map[intent].append(example.text.lower())

    def process(self, message: Message, **kwargs: Any) -> None:
        intent = {"name": self.parse(message.text), "confidence": 1.0}
        message.set("intent", intent, add_to_output=True)

    def parse(self, text: Text) -> Optional[Text]:
        _text = text.lower()

        def is_present(x):
            return x in _text

        for intent in self.intent_keyword_map.keys():
            if any(map(is_present, self.intent_keyword_map[intent])):
                return intent

        # If none of the keywords is in the text:
        return None

    def persist(self, file_name: Text, model_dir: Text) -> Dict[Text, Any]:
        """Persist this model into the passed directory.

        Return the metadata necessary to load the model again.
        """
        file_name = file_name + ".json"
        keyword_file = os.path.join(model_dir, file_name)
        utils.write_json_to_file(keyword_file, self.intent_keyword_map)

        return {"file": file_name}

    @classmethod
    def load(self,
             meta: Optional['Metadata'] = None,
             model_dir: Optional[Text] = None,
             model_metadata: 'Metadata' = None,
             cached_component: Optional['KeywordIntentClassifier'] = None,
             **kwargs: Any) -> 'KeywordIntentClassifier':

        file_name = meta.get("file")
        if file_name is not None:
            keyword_file = os.path.join(model_dir, file_name)
            if os.path.exists(keyword_file):
                self.intent_keyword_map = utils.read_json_file(keyword_file)
            else:
                logger.warning("Failed to load IntentKeywordClassifier, maybe {}"
                               "does not exist.".format(keyword_file))
        return self()
