from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import io
import json
import re
import warnings

from typing import Optional

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
    # type: ([str]) -> TrainingData
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


def load_luis_data(filename, tokenizer):
    # type: (str, Optional[Tokenizer]) -> TrainingData
    """Loads training data stored in the LUIS.ai data format."""

    warnings.warn(
        """LUIS data may not always be correctly imported because entity locations are specified by tokens.
        If you use a tokenizer which behaves differently from LUIS's your entities might not be correct""")
    if not tokenizer:
        raise ValueError("Can not load luis data without a specified tokenizer " +
                         "(e.g. using the configuration value `luis_data_tokenizer`)")

    intent_examples = []
    entity_examples = []
    common_examples = []

    with io.open(filename, encoding="utf-8-sig") as f:
        data = json.loads(f.read())
    for s in data["utterances"]:
        text = s.get("text")
        tokens = [t for t in tokenizer.tokenize(text)]
        intent = s.get("intent")
        entities = []
        for e in s.get("entities") or []:
            i, ii = e["startPos"], e["endPos"] + 1
            _regex = u"\s*".join([re.escape(s) for s in tokens[i:ii]])
            expr = re.compile(_regex)
            m = expr.search(text)
            start, end = m.start(), m.end()
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
    # type: (str) -> TrainingData
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
        intent = intents[0] if intents else None

        entities = [e for e in entities if ("start" in e and "end" in e)]
        for e in entities:
            e["value"] = e["value"][1:-1]

        if intent and entities:
            common_examples.append({"text": text, "intent": intent, "entities": entities})
        elif intent:
            intent_examples.append({"text": text, "intent": intent})
        elif entities:
            entity_examples.append({"text": text, "intent": intent, "entities": entities})
    return TrainingData(intent_examples, entity_examples, common_examples)


def load_rasa_data(filename):
    # type: (str) -> TrainingData
    """Loads training data stored in the rasa NLU data format."""

    with io.open(filename, encoding="utf-8-sig") as f:
        data = json.loads(f.read())
    common = data['rasa_nlu_data'].get("common_examples", list())
    intent = data['rasa_nlu_data'].get("intent_examples", list())
    entity = data['rasa_nlu_data'].get("entity_examples", list())

    return TrainingData(intent, entity, common)


def guess_format(files):
    # type: ([str]) -> str
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
    # type: (str) -> [str]
    """Lists all data files of the resource name (might be a file or directory)."""

    try:
        return utils.recursively_find_files(resource_name)
    except ValueError as e:
        raise ValueError("Invalid training data file / folder specified. " + e.message)


def load_data(resource_name, language, luis_data_tokenizer=None, fformat=None):
    # type: (str, str, Optional[Tokenizer], Optional[str]) -> TrainingData
    """Loads training data from disk. If no format is provided, the format will be guessed based on the files."""

    files = resolve_data_files(resource_name)

    if not fformat:
        fformat = guess_format(files)

    if fformat == LUIS_FILE_FORMAT:
        from rasa_nlu.tokenizers import tokenizer_from_name
        tokenizer = tokenizer_from_name(luis_data_tokenizer, language)
        return load_luis_data(files[0], tokenizer)
    elif fformat == WIT_FILE_FORMAT:
        return load_wit_data(files[0])
    elif fformat == API_FILE_FORMAT:
        return load_api_data(files)
    elif fformat == RASA_FILE_FORMAT:
        return load_rasa_data(files[0])
    else:
        raise ValueError("unknown training file format : {0}".format(fformat))
