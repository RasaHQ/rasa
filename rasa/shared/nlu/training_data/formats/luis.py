import logging
from typing import Any, Dict, List, Text

from rasa.shared.nlu.constants import (
    TEXT,
    INTENT,
    ENTITIES,
    ENTITY_ATTRIBUTE_TYPE,
    ENTITY_ATTRIBUTE_ROLE,
    ENTITY_ATTRIBUTE_VALUE,
    ENTITY_ATTRIBUTE_START,
    ENTITY_ATTRIBUTE_END,
    NO_ENTITY_TAG,
)
from rasa.shared.nlu.training_data.formats.readerwriter import JsonTrainingDataReader
import rasa.shared.utils.io

from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.message import Message

logger = logging.getLogger(__name__)


class LuisReader(JsonTrainingDataReader):
    """Reads LUIS training data."""

    @staticmethod
    def _extract_regex_features(js: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        regex_features: List[Dict[Text, Any]] = []

        for r in js.get("regex_features", []):
            if r.get("activated", False):
                regex_features.append(
                    {"name": r.get("name"), "pattern": r.get("pattern")}
                )

        # LUIS removed `regex_features` and exports regular expressions in `regex_entities` now:
        # https://stackoverflow.com/questions/48170631/what-happened-to-the-regex-features
        for r in js.get("regex_entities", []):
            regex_features.append(
                {"name": r.get("name"), "pattern": r.get("regexPattern")}
            )

        return regex_features

    def read_from_json(self, js: Dict[Text, Any], **kwargs: Any) -> "TrainingData":
        """Loads training data stored in the LUIS.ai data format."""
        training_examples = []

        max_tested_luis_schema_version = 7
        major_version = int(js["luis_schema_version"].split(".")[0])
        if major_version > max_tested_luis_schema_version:
            rasa.shared.utils.io.raise_warning(
                f"Your luis data schema version {js['luis_schema_version']} "
                f"is higher than 7.x.x. "
                f"Training may not be performed correctly. "
            )

        for s in js["utterances"]:
            text = s.get("text")
            intent = s.get("intent")
            entities = []
            for e in s.get("entities") or []:
                start, end = e["startPos"], e["endPos"] + 1
                val = text[start:end]

                entities.append(
                    {
                        ENTITY_ATTRIBUTE_TYPE: e.get("entity"),
                        ENTITY_ATTRIBUTE_VALUE: val,
                        ENTITY_ATTRIBUTE_START: start,
                        ENTITY_ATTRIBUTE_END: end,
                        ENTITY_ATTRIBUTE_ROLE: e.get("role", NO_ENTITY_TAG),
                    }
                )

            data = {ENTITIES: entities}
            if intent:
                data[INTENT] = intent
            data[TEXT] = text

            training_examples.append(Message(data=data))

        return TrainingData(
            training_examples, regex_features=self._extract_regex_features(js)
        )
