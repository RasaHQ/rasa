import logging
import typing
from collections import defaultdict
from typing import Any, Dict, Text

from rasa.constants import DOCS_URL_TRAINING_DATA_NLU
from rasa.nlu.training_data.formats.readerwriter import (
    JsonTrainingDataReader,
    TrainingDataWriter,
)
from rasa.nlu.training_data.util import transform_entity_synonyms
from rasa.nlu.utils import json_to_string
from rasa.utils.common import raise_warning

if typing.TYPE_CHECKING:
    from rasa.nlu.training_data import Message, TrainingData

logger = logging.getLogger(__name__)


class RasaReader(JsonTrainingDataReader):
    def read_from_json(self, js: Dict[Text, Any], **_) -> "TrainingData":
        """Loads training data stored in the rasa NLU data format."""
        from rasa.nlu.training_data import Message, TrainingData
        import rasa.nlu.schemas.data_schema as schema
        import rasa.utils.validation as validation_utils

        validation_utils.validate_training_data(js, schema.rasa_nlu_data_schema())

        data = js["rasa_nlu_data"]
        common_examples = data.get("common_examples", [])
        intent_examples = data.get("intent_examples", [])
        entity_examples = data.get("entity_examples", [])
        entity_synonyms = data.get("entity_synonyms", [])
        regex_features = data.get("regex_features", [])
        lookup_tables = data.get("lookup_tables", [])

        entity_synonyms = transform_entity_synonyms(entity_synonyms)

        if intent_examples or entity_examples:
            raise_warning(
                "Your rasa data "
                "contains 'intent_examples' "
                "or 'entity_examples' which will be "
                "removed in the future. Consider "
                "putting all your examples "
                "into the 'common_examples' section.",
                FutureWarning,
                docs=DOCS_URL_TRAINING_DATA_NLU,
            )

        all_examples = common_examples + intent_examples + entity_examples
        training_examples = []
        for ex in all_examples:
            msg = Message.build(ex["text"], ex.get("intent"), ex.get("entities"))
            training_examples.append(msg)

        return TrainingData(
            training_examples, entity_synonyms, regex_features, lookup_tables
        )


class RasaWriter(TrainingDataWriter):
    def dumps(self, training_data: "TrainingData", **kwargs) -> Text:
        """Writes Training Data to a string in json format."""

        js_entity_synonyms = defaultdict(list)
        for k, v in training_data.entity_synonyms.items():
            if k != v:
                js_entity_synonyms[v].append(k)

        formatted_synonyms = [
            {"value": value, "synonyms": syns}
            for value, syns in js_entity_synonyms.items()
        ]

        formatted_examples = [
            example.as_dict_nlu() for example in training_data.training_examples
        ]

        return json_to_string(
            {
                "rasa_nlu_data": {
                    "common_examples": formatted_examples,
                    "regex_features": training_data.regex_features,
                    "lookup_tables": training_data.lookup_tables,
                    "entity_synonyms": formatted_synonyms,
                }
            },
            **kwargs,
        )
