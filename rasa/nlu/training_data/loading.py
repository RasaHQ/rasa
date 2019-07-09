import json
import logging
import os

import requests
import typing
from typing import Optional, Text

import rasa.utils.io
from rasa.nlu import utils
from rasa.nlu.training_data.formats import markdown
from rasa.nlu.training_data.formats.dialogflow import (
    DIALOGFLOW_AGENT,
    DIALOGFLOW_ENTITIES,
    DIALOGFLOW_ENTITY_ENTRIES,
    DIALOGFLOW_INTENT,
    DIALOGFLOW_INTENT_EXAMPLES,
    DIALOGFLOW_PACKAGE,
)
from rasa.utils.endpoints import EndpointConfig
import rasa.utils.io as io_utils

if typing.TYPE_CHECKING:
    from rasa.nlu.training_data import TrainingData
    from rasa.nlu.training_data.formats.readerwriter import TrainingDataReader

logger = logging.getLogger(__name__)

# Different supported file formats and their identifier
WIT = "wit"
LUIS = "luis"
RASA = "rasa_nlu"
MARKDOWN = "md"
UNK = "unk"
DIALOGFLOW_RELEVANT = {DIALOGFLOW_ENTITIES, DIALOGFLOW_INTENT}

_markdown_section_markers = ["## {}:".format(s) for s in markdown.available_sections]
_json_format_heuristics = {
    WIT: lambda js, fn: "data" in js and isinstance(js.get("data"), list),
    LUIS: lambda js, fn: "luis_schema_version" in js,
    RASA: lambda js, fn: "rasa_nlu_data" in js,
    DIALOGFLOW_AGENT: lambda js, fn: "supportedLanguages" in js,
    DIALOGFLOW_PACKAGE: lambda js, fn: "version" in js and len(js) == 1,
    DIALOGFLOW_INTENT: lambda js, fn: "responses" in js,
    DIALOGFLOW_ENTITIES: lambda js, fn: "isEnum" in js,
    DIALOGFLOW_INTENT_EXAMPLES: lambda js, fn: "_usersays_" in fn,
    DIALOGFLOW_ENTITY_ENTRIES: lambda js, fn: "_entries_" in fn,
}


def load_data(resource_name: Text, language: Optional[Text] = "en") -> "TrainingData":
    """Load training data from disk.

    Merges them if loaded from disk and multiple files are found."""
    from rasa.nlu.training_data import TrainingData

    if not os.path.exists(resource_name):
        raise ValueError("File '{}' does not exist.".format(resource_name))

    files = utils.list_files(resource_name)
    data_sets = [_load(f, language) for f in files]
    data_sets = [ds for ds in data_sets if ds]
    if len(data_sets) == 0:
        training_data = TrainingData()
    elif len(data_sets) == 1:
        training_data = data_sets[0]
    else:
        training_data = data_sets[0].merge(*data_sets[1:])

    return training_data


async def load_data_from_endpoint(
    data_endpoint: EndpointConfig, language: Optional[Text] = "en"
) -> "TrainingData":
    """Load training data from a URL."""

    if not utils.is_url(data_endpoint.url):
        raise requests.exceptions.InvalidURL(data_endpoint.url)
    try:
        response = await data_endpoint.request("get")
        response.raise_for_status()
        temp_data_file = io_utils.create_temporary_file(response.content, mode="w+b")
        training_data = _load(temp_data_file, language)

        return training_data
    except Exception as e:
        logger.warning("Could not retrieve training data from URL:\n{}".format(e))


def _reader_factory(fformat: Text) -> Optional["TrainingDataReader"]:
    """Generates the appropriate reader class based on the file format."""
    from rasa.nlu.training_data.formats import (
        MarkdownReader,
        WitReader,
        LuisReader,
        RasaReader,
        DialogflowReader,
    )

    reader = None
    if fformat == LUIS:
        reader = LuisReader()
    elif fformat == WIT:
        reader = WitReader()
    elif fformat in DIALOGFLOW_RELEVANT:
        reader = DialogflowReader()
    elif fformat == RASA:
        reader = RasaReader()
    elif fformat == MARKDOWN:
        reader = MarkdownReader()
    return reader


def _load(filename: Text, language: Optional[Text] = "en") -> Optional["TrainingData"]:
    """Loads a single training data file from disk."""

    fformat = guess_format(filename)
    if fformat == UNK:
        raise ValueError("Unknown data format for file '{}'.".format(filename))

    logger.info("Training data format of '{}' is '{}'.".format(filename, fformat))
    reader = _reader_factory(fformat)

    if reader:
        return reader.read(filename, language=language, fformat=fformat)
    else:
        return None


def guess_format(filename: Text) -> Text:
    """Applies heuristics to guess the data format of a file.

    Args:
        filename: file whose type should be guessed

    Returns:
        Guessed file format.
    """
    guess = UNK

    content = ""
    try:
        content = rasa.utils.io.read_file(filename)
        js = json.loads(content)
    except ValueError:
        if any([marker in content for marker in _markdown_section_markers]):
            guess = MARKDOWN
    else:
        for fformat, format_heuristic in _json_format_heuristics.items():
            if format_heuristic(js, filename):
                guess = fformat
                break

    return guess


def _guess_format(filename: Text) -> Text:
    logger.warning(
        "Using '_guess_format()' is deprecated since Rasa 1.1.5. "
        "Please use 'guess_format()' instead."
    )
    return guess_format(filename)
