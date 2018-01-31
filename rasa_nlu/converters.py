from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Text

from rasa_nlu.training_data import TrainingData, Message, transform_entity_synonyms
from rasa_nlu import utils

logger = logging.getLogger(__name__)


# TODO: extract into DialogflowReader
def load_dialogflow_data(files, language):
    # type: (List[Text]) -> TrainingData
    """Loads training data stored in the Dialogflow data format."""

    training_examples = []
    entity_synonyms = {}
    for filename in files:
        data = utils.read_json_file(filename)
        # Language specific extensions
        usersays_file_ext = '_usersays_{}.json'.format(language)
        synonyms_file_ext = '_entries_{}.json'.format(language)
        if filename.endswith(usersays_file_ext):
            synonyms_filename = filename.replace(usersays_file_ext, '.json')
            root_f_data = utils.read_json_file(synonyms_filename)
            intent = root_f_data.get("name")

            for s in data:
                text = "".join([chunk["text"] for chunk in s.get("data")])
                # add entities to each token, if available
                entities = []
                for e in [chunk
                          for chunk in s.get("data")
                          if "alias" in chunk or "meta" in chunk]:
                    start = text.find(e["text"])
                    end = start + len(e["text"])
                    val = text[start:end]
                    entity_type = e["alias"] if "alias" in e else e["meta"]
                    if entity_type != u'@sys.ignore':
                        entities.append(
                            {
                                "entity": entity_type,
                                "value": val,
                                "start": start,
                                "end": end
                            }
                        )
                data = {}
                if intent:
                    data["intent"] = intent
                if entities is not None:
                    data["entities"] = entities
                training_examples.append(Message(text, data))

        elif filename.endswith(synonyms_file_ext):
            # create synonyms dictionary
            for entry in data:
                if "value" in entry and "synonyms" in entry:
                    for synonym in entry["synonyms"]:
                        entity_synonyms[synonym] = entry["value"]
    return TrainingData(training_examples, entity_synonyms)



# TODO: Extract into RasaJsonReader
def rasa_nlu_data_schema():
    training_example_schema = {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "intent": {"type": "string"},
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "start": {"type": "number"},
                        "end": {"type": "number"},
                        "value": {"type": "string"},
                        "entity": {"type": "string"}
                    },
                    "required": ["start", "end", "entity"]
                }
            }
        },
        "required": ["text"]
    }

    regex_feature_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "pattern": {"type": "string"},
        }
    }

    return {
        "type": "object",
        "properties": {
            "rasa_nlu_data": {
                "type": "object",
                "properties": {
                    "regex_features": {
                        "type": "array",
                        "items": regex_feature_schema
                    },
                    "common_examples": {
                        "type": "array",
                        "items": training_example_schema
                    },
                    "intent_examples": {
                        "type": "array",
                        "items": training_example_schema
                    },
                    "entity_examples": {
                        "type": "array",
                        "items": training_example_schema
                    }
                }
            }
        },
        "additionalProperties": False
    }

# TODO: Extract into RASAJsonReader
def validate_rasa_nlu_data(data):
    # type: (Dict[Text, Any]) -> None
    """Validate rasa training data format to ensure proper training.

    Raises exception on failure."""
    from jsonschema import validate
    from jsonschema import ValidationError

    try:
        validate(data, rasa_nlu_data_schema())
    except ValidationError as e:
        e.message += (". Failed to validate training data, make sure your data "
                      "is valid. For more information about the format visit "
                      "https://rasahq.github.io/rasa_nlu/dataformat.html")
        raise e

# TODO: Extract into DataHandler
def load_rasa_data(filenames):
    # type: (List[Text]) -> TrainingData
    """Loads training data stored in the rasa NLU data format."""

    common = list()
    intent = list()
    entity = list()
    regex_features = list()
    synonyms = list()
    for filename in filenames:
        data = utils.read_json_file(filename)
        validate_rasa_nlu_data(data)

        common += data['rasa_nlu_data'].get("common_examples", list())
        intent += data['rasa_nlu_data'].get("intent_examples", list())
        entity += data['rasa_nlu_data'].get("entity_examples", list())
        regex_features += data['rasa_nlu_data'].get("regex_features", list())
        synonyms += data['rasa_nlu_data'].get("entity_synonyms", list())

    entity_synonyms = transform_entity_synonyms(synonyms)

    if intent or entity:
        logger.warn("DEPRECATION warning: Data file \"{}\" contains 'intent_examples' "
                    "or 'entity_examples' which will be "
                    "removed in the future. Consider putting all your examples "
                    "into the 'common_examples' section.".format(filename))

    all_examples = common + intent + entity
    training_examples = []
    for e in all_examples:
        data = e.copy()
        if "text" in data:
            del data["text"]
        training_examples.append(Message(e["text"], data))

    return TrainingData(training_examples, entity_synonyms, regex_features)
