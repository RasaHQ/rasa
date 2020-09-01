import json
import logging
import os
import re
import typing
from typing import Optional, Text

import rasa.utils.io as io_utils
from rasa.nlu import utils
from rasa.nlu.training_data.formats.dialogflow import (
    DIALOGFLOW_AGENT,
    DIALOGFLOW_ENTITIES,
    DIALOGFLOW_ENTITY_ENTRIES,
    DIALOGFLOW_INTENT,
    DIALOGFLOW_INTENT_EXAMPLES,
    DIALOGFLOW_PACKAGE,
)
from rasa.utils.endpoints import EndpointConfig

if typing.TYPE_CHECKING:
    from rasa.nlu.training_data import TrainingData
    from rasa.nlu.training_data.formats.readerwriter import TrainingDataReader

logger = logging.getLogger(__name__)

# Different supported file formats and their identifier
WIT = "wit"
LUIS = "luis"
RASA = "rasa_nlu"
MARKDOWN = "md"
RASA_YAML = "rasa_yml"
UNK = "unk"
MARKDOWN_NLG = "nlg.md"
JSON = "json"
DIALOGFLOW_RELEVANT = {DIALOGFLOW_ENTITIES, DIALOGFLOW_INTENT}

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

# looks for pattern like:
# ##
# * intent/response_key
#   - response_text
_nlg_markdown_marker_regex = re.compile(r"##\s*.*\n\*[^:]*\/.*\n\s*\t*\-.*")


def load_data(resource_name: Text, language: Optional[Text] = "en") -> "TrainingData":
    """Load training data from disk.

    Merges them if loaded from disk and multiple files are found."""
    from rasa.nlu.training_data import TrainingData

    if not os.path.exists(resource_name):
        raise ValueError(f"File '{resource_name}' does not exist.")

    files = io_utils.list_files(resource_name)
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
    import requests

    if not utils.is_url(data_endpoint.url):
        raise requests.exceptions.InvalidURL(data_endpoint.url)
    try:
        response = await data_endpoint.request("get")
        response.raise_for_status()
        temp_data_file = io_utils.create_temporary_file(response.content, mode="w+b")
        training_data = _load(temp_data_file, language)

        return training_data
    except Exception as e:
        logger.warning(f"Could not retrieve training data from URL:\n{e}")


def _reader_factory(fformat: Text) -> Optional["TrainingDataReader"]:
    """Generates the appropriate reader class based on the file format."""
    from rasa.nlu.training_data.formats import (
        RasaYAMLReader,
        MarkdownReader,
        WitReader,
        LuisReader,
        RasaReader,
        DialogflowReader,
        NLGMarkdownReader,
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
    elif fformat == MARKDOWN_NLG:
        reader = NLGMarkdownReader()
    elif fformat == RASA_YAML:
        reader = RasaYAMLReader()
    return reader


def _load(filename: Text, language: Optional[Text] = "en") -> Optional["TrainingData"]:
    """Loads a single training data file from disk."""

    fformat = guess_format(filename)
    if fformat == UNK:
        raise ValueError(f"Unknown data format for file '{filename}'.")

    reader = _reader_factory(fformat)

    if reader:
        return reader.read(filename, language=language, fformat=fformat)
    else:
        return None


def _is_nlg_story_format(content: Text) -> bool:

    match = re.search(_nlg_markdown_marker_regex, content)
    if match:
        return True


def guess_format(filename: Text) -> Text:
    """Applies heuristics to guess the data format of a file.

    Args:
        filename: file whose type should be guessed

    Returns:
        Guessed file format.
    """
    from rasa.nlu.training_data.formats import RasaYAMLReader, markdown

    guess = UNK

    content = ""
    try:
        content = io_utils.read_file(filename)
        js = json.loads(content)
    except ValueError:
        if any(marker in content for marker in markdown.MARKDOWN_SECTION_MARKERS):
            guess = MARKDOWN
        elif _is_nlg_story_format(content):
            guess = MARKDOWN_NLG
        elif RasaYAMLReader.is_yaml_nlu_file(filename):
            guess = RASA_YAML
    else:
        for fformat, format_heuristic in _json_format_heuristics.items():
            if format_heuristic(js, filename):
                guess = fformat
                break

    logger.debug(f"Training data format of '{filename}' is '{guess}'.")

    return guess
