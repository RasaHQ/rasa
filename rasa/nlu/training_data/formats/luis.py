import logging
import typing
from typing import Any, Dict, Text

from rasa.nlu.training_data.formats.readerwriter import JsonTrainingDataReader

if typing.TYPE_CHECKING:
    from rasa.nlu.training_data import Message, TrainingData

logger = logging.getLogger(__name__)


class LuisReader(JsonTrainingDataReader):
    def is_luis_version_compatible(self, luisSchemaJson: Dict[Text, Any]):
        luisSchemaVersionChecked = 4
        version = int(luisSchemaJson["luis_schema_version"].split(".")[0])
        if version <= luisSchemaVersionChecked:
            return True
        else:
            return False

    def extract_regex_features_from_schema(self, luisSchemaJson: Dict[Text, Any]):
        regex_features = []
        for regex in luisSchemaJson.get("regex_features", []):
            if regex.get("activated", False):
                regex_features.append(
                    {"name": regex.get("name"), "pattern": regex.get("pattern")}
                )
        return regex_features

    def extract_entitie_fromUtter(self, utter, text):
        entities = []
        for entitie in utter.get("entities") or []:
            start, end = entitie["startPos"], entitie["endPos"] + 1
            val = text[start:end]
            entities.append(
                {"entity": entitie["entity"], "value": val, "start": start, "end": end}
            )

        return entities

    def extract_utters_from_schema(self, luisSchemaJson: Dict[Text, Any]):
        from rasa.nlu.training_data import Message

        training_examples = []
        for utter in luisSchemaJson["utterances"]:
            text = utter.get("text")
            entities = self.extract_entitie_fromUtter(utter, text)

            intent = utter.get("intent")
            data = {"entities": entities}
            if intent:
                data["intent"] = intent
            training_examples.append(Message(text, data))

        return training_examples

    def read_from_json(
        self, luisSchemaJson: Dict[Text, Any], **kwargs: Any
    ) -> "TrainingData":
        """Loads training data stored in the LUIS.ai data format."""
        from rasa.nlu.training_data import TrainingData

        if not self.is_luis_version_compatible(luisSchemaJson):
            logger.warning(
                "Your luis data schema version {} "
                "is higher than 4.x.x. "
                "Traning may not be performed correctly. "
                "".format(luisSchemaJson["luis_schema_version"])
            )

        regex_features = self.extract_regex_features_from_schema(luisSchemaJson)
        training_examples = self.extract_utters_from_schema(luisSchemaJson)

        return TrainingData(training_examples, regex_features=regex_features)
