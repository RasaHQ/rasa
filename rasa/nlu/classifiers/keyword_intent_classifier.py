import os
import logging
import typing
import re
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

        self.intent_keyword_map = intent_keyword_map or {}

        self.re_case_flag = (
            0 if self.component_config["case_sensitive"] else re.IGNORECASE
        )

    def train(
        self,
        training_data: "TrainingData",
        cfg: Optional["RasaNLUModelConfig"] = None,
        **kwargs: Any
    ) -> None:

        self.intent_keyword_map = {
            ex.text: ex.get("intent") for ex in training_data.training_examples
        }

        self._validate_keyword_map()

    def _validate_keyword_map(self):
        ambiguous_mappings = []
        for ex1, intent1 in self.intent_keyword_map.items():
            for ex2, intent2 in self.intent_keyword_map.items():
                if ex1 == ex2 and intent1 != intent2:
                    ambiguous_mappings.append((intent1, ex1))
                    ambiguous_mappings.append((intent2, ex2))
                    logger.warning(
                        "Keyword '{}' is an example of intent '{}' and"
                        " intent '{}', it will be removed from both.\n"
                        "Remove (one of) the conflicting examples for the"
                        " training data."
                        "".format(ex1, intent1, intent2)
                    )
                elif (
                    re.search(r"\b" + ex1 + r"\b", ex2, flags=self.re_case_flag)
                    and intent1 != intent2
                ):
                    ambiguous_mappings.append((intent1, ex1))
                    logger.warning(
                        "Keyword '{}' is an example of intent '{}',"
                        "but also a substring of '{}', which is an "
                        "example of intent '{}."
                        " '{}' will be removed from the list of keywords.\n"
                        "Remove (one of) the conflicting examples for the"
                        " training data."
                        "".format(ex1, intent1, ex2, intent2, ex1)
                    )
        for intent, example in ambiguous_mappings:
            self.intent_keyword_map.pop(example)
            logger.debug(
                "Removed keyword '{}' from intent '{}' because it matched"
                " another intent.".format(example, intent)
            )

    def process(self, message: Message, **kwargs: Any) -> None:
        intent_name = self._map_keyword_to_intent(message.text)
        confidence = 0.0 if intent_name is None else 1.0
        intent = {"name": intent_name, "confidence": confidence}
        if message.get("intent") is None or intent is not None:
            message.set("intent", intent, add_to_output=True)

    def _map_keyword_to_intent(self, text: Text) -> Optional[Text]:
        for example, intent in self.intent_keyword_map.items():
            if re.search(r"\b" + example + r"\b", text, flags=self.re_case_flag):
                logger.debug(
                    "KeywordClassifier matched keyword '{}' to"
                    " intent '{}'.".format(example, intent)
                )
                return intent
            else:
                logger.debug(
                    "KeywordClassifier did not find any keywords in " "the message."
                )
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
        return cls(meta, intent_keyword_map)
