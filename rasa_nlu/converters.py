from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import io
import json
import re
import warnings

import logging
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Text

from rasa_nlu import utils
from rasa_nlu.tokenizers import Tokenizer
from rasa_nlu.training_data import TrainingData

# Different supported file formats and their identifier
WIT_FILE_FORMAT = "wit"
API_FILE_FORMAT = "api"
LUIS_FILE_FORMAT = "luis"
RASA_FILE_FORMAT = "rasa_nlu"
UNK_FILE_FORMAT = "unk"


def load_api_data(files):
    # type: (List[Text]) -> TrainingData
    """Loads training data stored in the API.ai data format."""

    intent_examples = []
    entity_examples = []
    common_examples = []
    entity_synonyms = {}
    for filename in files:
        with io.open(filename, encoding="utf-8-sig") as f:
            data = json.loads(f.read())
        # get only intents, skip the rest. The property name is the target class
        if "userSays" in data:
            intent = data.get("name")
            for s in data["userSays"]:
                text = "".join([chunk["text"] for chunk in s.get("data")])
                # add entities to each token, if available
                entities = []
                for e in [chunk for chunk in s.get("data") if "alias" in chunk or "meta" in chunk]:
                    start = text.find(e["text"])
                    end = start + len(e["text"])
                    val = text[start:end]
                    entities.append(
                        {
                            "entity": e["alias"] if "alias" in e else e["meta"],
                            "value": val,
                            "start": start,
                            "end": end
                        }
                    )

                if intent and entities:
                    common_examples.append({"text": text, "intent": intent, "entities": entities})
                elif intent:
                    intent_examples.append({"text": text, "intent": intent})
                elif entities:
                    entity_examples.append({"text": text, "intent": intent, "entities": entities})

        # create synonyms dictionary
        if "name" in data and "entries" in data:
            for entry in data["entries"]:
                if "value" in entry and "synonyms" in entry:
                    for synonym in entry["synonyms"]:
                        entity_synonyms[synonym] = entry["value"]
    return TrainingData(intent_examples, entity_examples, common_examples, entity_synonyms)


def load_luis_data(filename):
    # type: (Text) -> TrainingData
    """Loads training data stored in the LUIS.ai data format."""

    intent_examples = []
    entity_examples = []
    common_examples = []

    with io.open(filename, encoding="utf-8-sig") as f:
        data = json.loads(f.read())

    # Simple check to ensure we support this luis data schema version
    if not data["luis_schema_version"].startswith("2"):
        raise Exception("Invalid luis data schema version {}, should be 2.x.x. ".format(data["luis_schema_version"]) +
                        "Make sure to use the latest luis version (e.g. by downloading your data again).")

    for s in data["utterances"]:
        text = s.get("text")
        intent = s.get("intent")
        entities = []
        for e in s.get("entities") or []:
            start, end = e["startPos"], e["endPos"] + 1
            val = text[start:end]
            entities.append({"entity": e["entity"], "value": val, "start": start, "end": end})

        if intent and entities:
            common_examples.append({"text": text, "intent": intent, "entities": entities})
        elif intent:
            intent_examples.append({"text": text, "intent": intent})
        elif entities:
            entity_examples.append({"text": text, "intent": intent, "entities": entities})
    return TrainingData(intent_examples, entity_examples, common_examples)


def load_wit_data(filename):
    # type: (Text) -> TrainingData
    """Loads training data stored in the WIT.ai data format."""

    intent_examples = []
    entity_examples = []
    common_examples = []

    with io.open(filename, encoding="utf-8-sig") as f:
        data = json.loads(f.read())
    for s in data["data"]:
        entities = s.get("entities")
        if entities is None:
            continue
        text = s.get("text")
        intents = [e["value"] for e in entities if e["entity"] == 'intent']
        intent = intents[0].strip("\"") if intents else None

        entities = [e for e in entities if ("start" in e and "end" in e and e["entity"] != 'intent')]
        for e in entities:
            e["value"] = e["value"].strip("\"")    # for some reason wit adds additional quotes around entity values

        if intent and entities:
            common_examples.append({"text": text, "intent": intent, "entities": entities})
        elif intent:
            intent_examples.append({"text": text, "intent": intent})
        elif entities:
            entity_examples.append({"text": text, "intent": intent, "entities": entities})
    return TrainingData(intent_examples, entity_examples, common_examples)


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

    return {
        "type": "object",
        "properties": {
            "rasa_nlu_data": {
                "type": "object",
                "properties": {
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


def validate_rasa_nlu_data(data):
    # type: (Dict[Text, Any]) -> None
    """Validate rasa training data format to ensure proper training. Raises exception on failure."""
    from jsonschema import validate
    from jsonschema import ValidationError

    try:
        validate(data, rasa_nlu_data_schema())
    except ValidationError as e:
        e.message += \
            ". Failed to validate training data, make sure your data is valid. " + \
            "For more information about the format visit " + \
            "https://rasa-nlu.readthedocs.io/en/latest/dataformat.html"
        raise e


def load_rasa_data(filename):
    # type: (Text) -> TrainingData
    """Loads training data stored in the rasa NLU data format."""

    with io.open(filename, encoding="utf-8-sig") as f:
        data = json.loads(f.read())
    validate_rasa_nlu_data(data)
    common = data['rasa_nlu_data'].get("common_examples", list())
    intent = data['rasa_nlu_data'].get("intent_examples", list())
    entity = data['rasa_nlu_data'].get("entity_examples", list())

    return TrainingData(intent, entity, common)


def guess_format(files):
    # type: (List[Text]) -> Text
    """Given a set of files, tries to guess which data format is used."""

    for filename in files:
        with io.open(filename, encoding="utf-8-sig") as f:
            file_data = json.loads(f.read())
        if "data" in file_data and type(file_data.get("data")) is list:
            return WIT_FILE_FORMAT
        elif "luis_schema_version" in file_data:
            return LUIS_FILE_FORMAT
        elif "userSays" in file_data:
            return API_FILE_FORMAT
        elif "rasa_nlu_data" in file_data:
            return RASA_FILE_FORMAT

    return UNK_FILE_FORMAT


def resolve_data_files(resource_name):
    # type: (Text) -> List[Text]
    """Lists all data files of the resource name (might be a file or directory)."""

    try:
        return utils.recursively_find_files(resource_name)
    except ValueError as e:
        raise ValueError("Invalid training data file / folder specified. {}".format(e))


def load_data(resource_name, fformat=None):
    # type: (Text, Optional[Text]) -> TrainingData
    """Loads training data from disk. If no format is provided, the format will be guessed based on the files."""

    files = resolve_data_files(resource_name)

    if not fformat:
        fformat = guess_format(files)

    logging.info("Training data format at {} is {}".format(resource_name, fformat))

    if fformat == LUIS_FILE_FORMAT:
        return load_luis_data(files[0])
    elif fformat == WIT_FILE_FORMAT:
        return load_wit_data(files[0])
    elif fformat == API_FILE_FORMAT:
        return load_api_data(files)
    elif fformat == RASA_FILE_FORMAT:
        return load_rasa_data(files[0])
    else:
        raise ValueError("unknown training file format : {} for file {}".format(fformat, resource_name))
