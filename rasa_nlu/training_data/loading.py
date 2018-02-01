from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import json

from rasa_nlu import utils
from rasa_nlu.training_data.formats import MarkdownReader, WitReader, LuisReader, RasaReader, DialogflowReader
from rasa_nlu.training_data import TrainingData
from rasa_nlu.training_data.formats.dialogflow import DIALOGFLOW_AGENT, DIALOGFLOW_PACKAGE, DIALOGFLOW_INTENT, \
    DIALOGFLOW_ENTITIES, DIALOGFLOW_ENTITY_ENTRIES, DIALOGFLOW_INTENT_EXAMPLES

logger = logging.getLogger(__name__)

# Different supported file formats and their identifier
WIT = "wit"
LUIS = "luis"
RASA = "rasa_nlu"
UNK = "unk"
MARKDOWN = "md"

_json_format_heuristics = {
    WIT: lambda js, fn: "data" in js and isinstance(js.get("data"), list),
    LUIS: lambda js, fn: "luis_schema_version" in js,
    RASA: lambda js, fn: "rasa_nlu_data" in js,
    DIALOGFLOW_AGENT: lambda js, fn: "supportedLanguages" in js,
    DIALOGFLOW_PACKAGE: lambda js, fn: "version" in js and len(js) == 1,
    DIALOGFLOW_INTENT: lambda js, fn: "responses" in js,
    DIALOGFLOW_ENTITIES: lambda js, fn: "isEnum" in js,
    DIALOGFLOW_INTENT_EXAMPLES: lambda js, fn: "_usersays_" in fn,
    DIALOGFLOW_ENTITY_ENTRIES: lambda js, fn: "_entries_" in fn
}


def _from_dialogflow_file(filename, language, fformat):
    if fformat in {DIALOGFLOW_INTENT, DIALOGFLOW_ENTITIES}:
        return DialogflowReader().read(filename, language=language, fformat=fformat)
    else:
        return None


def from_rasa_file(filename, **kwargs):
    return RasaReader().read(filename, **kwargs)


def from_wit_file(filename, **kwargs):
    return WitReader().read(filename, **kwargs)


def from_luis_file(filename, **kwargs):
    return LuisReader().read(filename, **kwargs)


def from_markdown_file(filename, **kwargs):
    return MarkdownReader().read(filename, **kwargs)


def load_data(resource_name, language='en'):
    # type: (Text, Optional[Text]) -> TrainingData
    """Loads training data from disk and merges them if multiple files are found."""

    files = utils.recursively_find_files(resource_name)
    data_sets = [_load(f, language) for f in files]
    # Dialogflow has files that we don't read directly, these return None
    data_sets = [ds for ds in data_sets if ds]
    if len(data_sets) == 0:
        return TrainingData()
    elif len(data_sets) == 1:
        return data_sets[0]
    else:
        return data_sets[0].merge(*data_sets[1:])


def _load(filename, language='en'):
    """Loads a single training data file from disk."""

    fformat = _guess_format(filename)

    logger.info("Training data format of {} is {}".format(filename, fformat))

    if fformat == LUIS:
        return from_luis_file(filename)
    elif fformat == WIT:
        return from_wit_file(filename)
    elif fformat.startswith("dialogflow"):
        return _from_dialogflow_file(filename, language, fformat)
    elif fformat == RASA:
        return from_rasa_file(filename)
    elif fformat == MARKDOWN:
        return from_markdown_file(filename)
    else:
        raise ValueError("unknown training file format : {} for "
                         "file {}".format(fformat, filename))


def _guess_format(filename):
    # type: (Text) -> Text
    """Applies heuristics to guess the data format of a file."""
    guess = UNK
    content = utils.read_file(filename)
    try:
        js = json.loads(content)
    except ValueError:
        if "## intent:" in content:
            guess = MARKDOWN
    else:
        for fformat, format_heuristic in _json_format_heuristics.items():
            if format_heuristic(js, filename):
                guess = fformat

    return guess
