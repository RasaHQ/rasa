import json
import logging
import os

import requests
import typing
from rasa.nlu.training_data.formats import SupportedFormats
from typing import Optional, Text
from rasa.nlu.training_data.reader_factory import ReaderFactory

from rasa.nlu import utils
from rasa.nlu.training_data.formats import markdown
from rasa.utils.endpoints import EndpointConfig
import rasa.utils.io as io_utils

if typing.TYPE_CHECKING:
    from rasa.nlu.training_data.training_data import TrainingData

from rasa.nlu.training_data.formats.dialogflow import (
    DIALOGFLOW_AGENT,
    DIALOGFLOW_ENTITIES,
    DIALOGFLOW_ENTITY_ENTRIES,
    DIALOGFLOW_INTENT,
    DIALOGFLOW_INTENT_EXAMPLES,
    DIALOGFLOW_PACKAGE,
)

logger = logging.getLogger(__name__)

markdown_section_markers = ["## {}:".format(s) for s in markdown.available_sections]
json_format_heuristics = {
    SupportedFormats.WIT: lambda js, fn: "data" in js
    and isinstance(js.get("data"), list),
    SupportedFormats.LUIS: lambda js, fn: "luis_schema_version" in js,
    SupportedFormats.RASA: lambda js, fn: "rasa_nlu_data" in js,
    DIALOGFLOW_AGENT: lambda js, fn: "supportedLanguages" in js,
    DIALOGFLOW_PACKAGE: lambda js, fn: "version" in js and len(js) == 1,
    DIALOGFLOW_INTENT: lambda js, fn: "responses" in js,
    DIALOGFLOW_ENTITIES: lambda js, fn: "isEnum" in js,
    DIALOGFLOW_INTENT_EXAMPLES: lambda js, fn: "_usersays_" in fn,
    DIALOGFLOW_ENTITY_ENTRIES: lambda js, fn: "_entries_" in fn,
}


class DataManager:
    @staticmethod
    def load_data(
        resource_name: Text, language: Optional[Text] = "en"
    ) -> "TrainingData":
        """Load training data from disk.

        Merges them if loaded from disk and multiple files are found."""
        from rasa.nlu.training_data.training_data import TrainingData

        if not os.path.exists(resource_name):
            raise ValueError("File '{}' does not exist.".format(resource_name))

        files = io_utils.list_files(resource_name)
        data_sets = [DataManager.__load(f, language) for f in files]
        data_sets = [ds for ds in data_sets if ds]
        if len(data_sets) == 0:
            training_data = TrainingData()
        elif len(data_sets) == 1:
            training_data = data_sets[0]
        else:
            training_data = data_sets[0].merge(*data_sets[1:])

        return training_data

    @staticmethod
    async def load_data_from_endpoint(
        data_endpoint: EndpointConfig, language: Optional[Text] = "en"
    ) -> Optional["TrainingData"]:
        """Load training data from a URL."""

        if not utils.is_url(data_endpoint.url):
            raise requests.exceptions.InvalidURL(data_endpoint.url)
        try:
            response = await data_endpoint.request("get")
            response.raise_for_status()
            temp_data_file = io_utils.create_temporary_file(
                response.content, mode="w+b"
            )
            training_data = DataManager.__load(temp_data_file, language)

            return training_data
        except Exception as e:
            logger.warning("Could not retrieve training data from URL:\n{}".format(e))

    @staticmethod
    def guess_format(filename: Text) -> Text:
        """Applies heuristics to guess the data format of a file.

        Args:
            filename: file whose type should be guessed

        Returns:
            Guessed file format.
        """
        guess = SupportedFormats.UNK

        content = ""
        try:
            content = io_utils.read_file(filename)
            js = json.loads(content)
        except ValueError:
            if any([marker in content for marker in markdown_section_markers]):
                guess = SupportedFormats.MARKDOWN
        else:
            for file_format, format_heuristic in json_format_heuristics.items():
                if format_heuristic(js, filename):
                    guess = file_format
                    break

        return guess

    @staticmethod
    def __load(
        filename: Text, language: Optional[Text] = "en"
    ) -> Optional["TrainingData"]:
        """ Helper method for loading file from disk."""

        file_format = DataManager.guess_format(filename)
        if file_format == SupportedFormats.UNK:
            raise ValueError("Unknown data format for file '{}'.".format(filename))

        logger.debug(
            "Training data format of '{}' is '{}'.".format(filename, file_format)
        )
        reader = ReaderFactory.get_reader(file_format)

        if reader:
            return reader.read(filename, language=language, file_format=file_format)
        else:
            return None
