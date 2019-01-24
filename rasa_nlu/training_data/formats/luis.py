import logging
import typing
from typing import Any, Dict, Text

from rasa_nlu.training_data.formats.readerwriter import JsonTrainingDataReader

if typing.TYPE_CHECKING:
    from rasa_nlu.training_data import Message, TrainingData

logger = logging.getLogger(__name__)


class LuisReader(JsonTrainingDataReader):

    def read_from_json(self,
                       js: Dict[Text, Any],
                       **kwargs: Any) -> 'TrainingData':
        """Loads training data stored in the LUIS.ai data format."""
        from rasa_nlu.training_data import Message, TrainingData

        training_examples = []
        regex_features = []

        # Simple check to ensure we support this luis data schema version
        if not js["luis_schema_version"].startswith("2"):
            raise Exception("Invalid luis data schema version {}, "
                            "should be 2.x.x. "
                            "Make sure to use the latest luis version "
                            "(e.g. by downloading your data again)."
                            "".format(js["luis_schema_version"]))

        for r in js.get("regex_features", []):
            if r.get("activated", False):
                regex_features.append({"name": r.get("name"),
                                       "pattern": r.get("pattern")})

        for s in js["utterances"]:
            text = s.get("text")
            intent = s.get("intent")
            entities = []
            for e in s.get("entities") or []:
                start, end = e["startPos"], e["endPos"] + 1
                val = text[start:end]
                entities.append({"entity": e["entity"],
                                 "value": val,
                                 "start": start,
                                 "end": end})

            data = {"entities": entities}
            if intent:
                data["intent"] = intent
            training_examples.append(Message(text, data))
        return TrainingData(training_examples, regex_features=regex_features)
