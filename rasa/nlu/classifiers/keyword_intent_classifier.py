import os
import logging
import copy
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

    defaults = {"case_sensitive": True}

    def __init__(
        self,
        component_config: Optional[Dict[Text, Any]] = None,
        intent_keyword_map: Optional[Dict] = None,
    ):

        super(KeywordIntentClassifier, self).__init__(component_config)

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

        for intent in training_data.intents:
            self.intent_keyword_map[intent] = [
                ex.text
                for ex in training_data.training_examples
                if ex.get("intent") == intent
            ]
        self._validate_keyword_map(self.intent_keyword_map)

    @staticmethod
    def _validate_keyword_map(keyword_map):
        for i, (intent1, ex1s) in enumerate(keyword_map.items()):
            for j, (intent2, ex2s) in enumerate(keyword_map.items()):
                if j > i:
                    comp_string1 = "\n".join(ex2s)
                    comp_string2 = "\n".join(ex2s)
                    for ex1 in ex1s:
                        if ex1 in comp_string2:
                            logger.warning("Keyword '{}' is an example of intent '{}',"
                                           "but also matches intent '{}"
                                           "".format(ex1, intent1, intent2))
                    for ex2 in ex2s:
                        if ex2 in comp_string1:
                            logger.warning("Keyword '{}' is an example of intent '{}',"
                                           "but also matches intent '{}"
                                           "".format(ex2, intent2, intent1))


    def process(self, message: Message, **kwargs: Any) -> None:
        intent_name = self._map_keyword_to_intent(message.text)
        if intent_name is not None:
            intent = {"name": intent_name, "confidence": 1.0}
            message.set("intent", intent, add_to_output=True)

    def _map_keyword_to_intent(self, text: Text) -> Optional[Text]:
        found_intents = []
        for intent, examples in self.intent_keyword_map.items():
            for example in examples:
                if self.component_config["case_sensitive"]:
                    if example in text:
                        found_intents.append(intent)
                else:
                    if example.lower() in text.lower():
                        found_intents.append(intent)
        if len(found_intents) == 0:
            return None
        elif len(found_intents) == 1:
            return found_intents[0]
        else:
            logger.debug(
                "KeywordClassifier found keywords for intents '{}',"
                "will classify message as having intent '{}'."
                "".format(found_intents, found_intents[0])
            )
            return found_intents[0]

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
        return cls(meta, intent_keyword_map)
