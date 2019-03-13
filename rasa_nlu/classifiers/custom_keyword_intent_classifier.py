from typing import Any, Dict, Optional, Text

from rasa_nlu.components import Component
from rasa_nlu.training_data import Message
import pickle
import os


class CustomKeywordIntentClassifier(Component):
    """Intent classifier using simple keyword matching.


    The classifier takes a list of keywords and associated intents as an input.
    A input sentence is checked for the keywords and the intent is returned.

    """

    name = "intent_classifier_custom_keyword"

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
                if example.data ['intent'] == intent:
                    self.intent_keyword_map[intent].append(example.text.lower())

    def process(self, message: Message, **kwargs: Any) -> None:
        intent = {"name": self.parse(message.text), "confidence": 1.0}
        message.set("intent", intent,
                    add_to_output=True)

    def parse(self, text: Text) -> Optional[Text]:
        _text = text.lower()

        def is_present(x):
            return x in _text

        for intent in self.intent_keyword_map.keys():
            if any(map(is_present, self.intent_keyword_map[intent])):
                return intent

        # If none of the keywords is in the text:
        return None

    def persist(self, model_dir: Text) -> Dict[Text, Any]:
        keyword_file = os.path.join(model_dir, "keys.p")
        pickle.dump(self.intent_keyword_map,  open(keyword_file, "wb"))

        return {"file": keyword_file}

    @classmethod
    def load(self,
             model_dir: Optional[Text] = None,
             model_metadata: Optional['Metadata'] = None,
             cached_component: Optional['Component'] = None,
             **kwargs: Any) -> 'CustomKeywordIntentClassifier':

        keyword_file = os.path.join(model_dir, "keys.p")

        self.intent_keyword_map = pickle.load(open(keyword_file, "rb"))

        return self()
