from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import io
import json
import logging

import os
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Text

from rasa_nlu import utils
from rasa_nlu.training_data import TrainingData, Message

logger = logging.getLogger(__name__)

# Different supported file formats and their identifier
WIT_FILE_FORMAT = "wit"
DIALOGFLOW_FILE_FORMAT = "dialogflow"
LUIS_FILE_FORMAT = "luis"
RASA_FILE_FORMAT = "rasa_nlu"
UNK_FILE_FORMAT = "unk"
MARKDOWN_FILE_FORMAT = "md"


def _read_json_from_file(filename):
    with io.open(filename, encoding="utf-8-sig") as f:
        try:
            return json.loads(f.read())
        except Exception as e:
            raise Exception("Failed to read json from '{}'. Error: "
                            "{}".format(os.path.abspath(filename), e))


def load_dialogflow_data(files, language):
    # type: (List[Text]) -> TrainingData
    """Loads training data stored in the Dialogflow data format."""

    training_examples = []
    entity_synonyms = {}
    for filename in files:
        data = _read_json_from_file(filename)
        # Language specific extensions
        usersays_file_ext = '_usersays_{}.json'.format(language)
        synonyms_file_ext = '_entries_{}.json'.format(language)
        if filename.endswith(usersays_file_ext):
            synonyms_filename = filename.replace(usersays_file_ext, '.json')
            root_f_data = _read_json_from_file(synonyms_filename)
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


def load_luis_data(filename):
    # type: (Text) -> TrainingData
    """Loads training data stored in the LUIS.ai data format."""

    training_examples = []
    regex_features = []

    data = _read_json_from_file(filename)

    # Simple check to ensure we support this luis data schema version
    if not data["luis_schema_version"].startswith("2"):
        raise Exception("Invalid luis data schema version {}, should be 2.x.x. "
                        "Make sure to use the latest luis version "
                        "(e.g. by downloading your data again)."
                        "".format(data["luis_schema_version"]))

    for r in data.get("regex_features", []):
        if r.get("activated", False):
            regex_features.append({"name": r.get("name"),
                                   "pattern": r.get("pattern")})

    for s in data["utterances"]:
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


def load_wit_data(filename):
    # type: (Text) -> TrainingData
    """Loads training data stored in the WIT.ai data format."""

    training_examples = []

    data = _read_json_from_file(filename)
    for s in data["data"]:
        entities = s.get("entities")
        if entities is None:
            continue
        text = s.get("text")
        intents = [e["value"] for e in entities if e["entity"] == 'intent']
        intent = intents[0].strip("\"") if intents else None

        entities = [e
                    for e in entities
                    if ("start" in e and "end" in e and
                        e["entity"] != 'intent')]
        for e in entities:
            # for some reason wit adds additional quotes around entity values
            e["value"] = e["value"].strip("\"")

        data = {}
        if intent:
            data["intent"] = intent
        if entities is not None:
            data["entities"] = entities
        training_examples.append(Message(text, data))
    return TrainingData(training_examples)


def load_markdown_data(filename):
    # type: (Text) -> TrainingData
    """Loads training data stored in markdown data format."""
    from rasa_nlu.utils.md_to_json import MarkdownToJson
    data = MarkdownToJson(filename)
    return TrainingData(data.common_examples,
                        get_entity_synonyms_dict(data.entity_synonyms))


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


def load_rasa_data(filename):
    # type: (Text) -> TrainingData
    """Loads training data stored in the rasa NLU data format."""

    data = _read_json_from_file(filename)
    validate_rasa_nlu_data(data)

    common = data['rasa_nlu_data'].get("common_examples", list())
    intent = data['rasa_nlu_data'].get("intent_examples", list())
    entity = data['rasa_nlu_data'].get("entity_examples", list())
    regex_features = data['rasa_nlu_data'].get("regex_features", list())
    synonyms = data['rasa_nlu_data'].get("entity_synonyms", list())

    entity_synonyms = get_entity_synonyms_dict(synonyms)

    if intent or entity:
        logger.warn("DEPRECATION warning: Data file contains 'intent_examples' "
                    "or 'entity_examples' which will be "
                    "removed in the future. Consider putting all your examples "
                    "into the 'common_examples' section.")

    all_examples = common + intent + entity
    training_examples = []
    for e in all_examples:
        data = e.copy()
        if "text" in data:
            del data["text"]
        training_examples.append(Message(e["text"], data))

    return TrainingData(training_examples, entity_synonyms, regex_features)


def get_entity_synonyms_dict(synonyms):
    # type: (List[Dict]) -> Dict
    """build entity_synonyms dictionary"""
    entity_synonyms = {}
    for s in synonyms:
        if "value" in s and "synonyms" in s:
            for synonym in s["synonyms"]:
                entity_synonyms[synonym] = s["value"]
    return entity_synonyms


def guess_format(files):
    # type: (List[Text]) -> Text
    """Given a set of files, tries to guess which data format is used."""

    for filename in files:
        with io.open(filename, encoding="utf-8-sig") as f:
            raw_data = ""
            try:
                raw_data = f.read()
                file_data = json.loads(raw_data)
                if "data" in file_data and type(file_data.get("data")) is list:
                    return WIT_FILE_FORMAT
                elif "luis_schema_version" in file_data:
                    return LUIS_FILE_FORMAT
                elif "supportedLanguages" in file_data:
                    return DIALOGFLOW_FILE_FORMAT
                elif "rasa_nlu_data" in file_data:
                    return RASA_FILE_FORMAT

            except ValueError:
                if "## intent:" in raw_data:
                    return MARKDOWN_FILE_FORMAT

    return UNK_FILE_FORMAT


def resolve_data_files(resource_name):
    # type: (Text) -> List[Text]
    """Lists all data files of the resource name

    (might be a file or directory)."""

    try:
        return utils.recursively_find_files(resource_name)
    except ValueError as e:
        raise ValueError("Invalid training data file / folder specified. "
                         "{}".format(e))


def load_data(resource_name, language='en', fformat=None):
    # type: (Text, Optional[Text]) -> TrainingData
    """Loads training data from disk.

    If no format is provided, the format will be guessed based on the files."""

    files = resolve_data_files(resource_name)

    if not fformat:
        fformat = guess_format(files)

    logger.info("Training data format at {} is {}".format(resource_name,
                                                          fformat))

    if fformat == LUIS_FILE_FORMAT:
        return load_luis_data(files[0])
    elif fformat == WIT_FILE_FORMAT:
        return load_wit_data(files[0])
    elif fformat == DIALOGFLOW_FILE_FORMAT:
        return load_dialogflow_data(files, language)
    elif fformat == RASA_FILE_FORMAT:
        return load_rasa_data(files[0])
    elif fformat == MARKDOWN_FILE_FORMAT:
        return load_markdown_data(files[0])
    else:
        raise ValueError("unknown training file format : {} for "
                         "file {}".format(fformat, resource_name))
