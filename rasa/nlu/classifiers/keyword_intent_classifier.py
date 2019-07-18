import os
import logging
import re
import typing
from typing import Any, Dict, Optional, Text

from rasa.nlu import utils
from rasa.nlu.components import Component
from rasa.nlu.training_data import Message

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from rasa.nlu.config import RasaNLUModelConfig
    from rasa.nlu.training_data import TrainingData
    from rasa.nlu.model import Metadata
    from rasa.nlu.training_data import Message


class KeywordIntentClassifier(Component):
    """Intent classifier using simple keyword matching.


    The classifier takes a list of keywords and associated intents as an input.
    A input sentence is checked for the keywords and the intent is returned.

    """

    provides = ["intent"]

    def __init__(self, intent_keyword_map: Optional[Dict] = None):

        if intent_keyword_map is None:
            self.intent_keyword_map = {}
        else:
            self.intent_keyword_map = intent_keyword_map

    def train(
        self,
        training_data: "TrainingData",
        cfg: Optional["RasaNLUModelConfig"] = None,
        **kwargs: Any
    ) -> None:
        self.intent_keyword_map = {}
        for intent in training_data.intents:
            self.intent_keyword_map[intent] = [
                ex.text
                for ex in training_data.training_examples
                if ex.get("intent") == intent
            ]

    def process(self, message: Message, **kwargs: Any) -> None:
        intent_name = self.map_keyword_to_intent(message.text)
        if intent_name is not None:
            intent = {"name": intent_name, "confidence": 1.0}
            message.set("intent", intent, add_to_output=True)

    def map_keyword_to_intent(self, text: Text) -> Optional[Text]:

        for intent, keywords in self.intent_keyword_map.items():
            for string in keywords:
                if re.search(r"\b" + string + r"\b", text):
                    return intent

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
    def load(
        cls,
        meta: Dict[Text, Any],
        model_dir: Optional[Text] = None,
        model_metadata: "Metadata" = None,
        cached_component: Optional["KeywordIntentClassifier"] = None,
        **kwargs: Any
    ) -> "KeywordIntentClassifier":

        if model_dir and meta.get("file"):
            file_name = meta.get("file")
            keyword_file = os.path.join(model_dir, file_name)
            if os.path.exists(keyword_file):
                intent_keyword_map = utils.read_json_file(keyword_file)
            else:
                logger.warning(
                    "Failed to load IntentKeywordClassifier, maybe "
                    "{} does not exist.".format(keyword_file)
                )
        return cls(intent_keyword_map)
