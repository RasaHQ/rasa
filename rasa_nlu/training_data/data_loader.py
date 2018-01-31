from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import json

from rasa_nlu import utils
from rasa_nlu.training_data.formats import MarkdownReader, WitReader, LuisReader
from rasa_nlu.training_data import TrainingData

logger = logging.getLogger(__name__)

# Different supported file formats and their identifier
WIT_FILE_FORMAT = "wit"
DIALOGFLOW_FILE_FORMAT = "dialogflow"
LUIS_FILE_FORMAT = "luis"
RASA_FILE_FORMAT = "rasa_nlu"
UNK_FILE_FORMAT = "unk"
MARKDOWN_FILE_FORMAT = "md"


class DataLoader(object):

    @classmethod
    def from_wit_file(cls, filename):
        cls._from_file(WitReader, filename)

    @classmethod
    def from_luis_file(cls, filename):
        cls._from_file(LuisReader, filename)

    @classmethod
    def from_markdown_file(cls, filename):
        return cls._from_file(MarkdownReader, filename)

    @classmethod
    def from_markdown_string(cls, s):
        return cls._from_string(MarkdownReader, s)

    @staticmethod
    def _from_file(reader_cls, filename, **kwargs):
        reader = reader_cls()
        return reader.read(filename, **kwargs)

    @staticmethod
    def _from_string(reader_cls, s, **kwargs):
        reader = reader_cls()
        return reader.reads(s, **kwargs)

    @staticmethod
    def _from_json(reader_cls, js, **kwargs):
        reader = reader_cls()
        return reader.read_from_json(js, **kwargs)

    @classmethod
    def load(cls, resource_name, language='en', fformat=None):
        # type: (Text, Optional[Text]) -> TrainingData
        """Loads training data from disk and merges them if multiple files are found."""

        files = utils.recursively_find_files(resource_name)
        data_sets = [cls._load(f, language, fformat) for f in files]
        if len(data_sets) == 0:
            return TrainingData()
        elif len(data_sets) == 1:
            return data_sets[0]
        else:
            return data_sets[0].merge(*data_sets[1:])

    @classmethod
    def _load(cls, filename, language='en', fformat=None):
        """Loads a single training data file from disk.

        Guesses the format if it's not provided."""
        if not fformat:
            fformat = cls._guess_format(filename)

        logger.info("Training data format of {} is {}".format(filename, fformat))

        if fformat == LUIS_FILE_FORMAT:
            return cls.from_luis_file(filename)
        elif fformat == WIT_FILE_FORMAT:
            return cls.from_wit_file(filename)
        elif fformat == DIALOGFLOW_FILE_FORMAT:
            return load_dialogflow_data(filename, language)
        elif fformat == RASA_FILE_FORMAT:
            return load_rasa_data(filename)
        elif fformat == MARKDOWN_FILE_FORMAT:
            return cls.from_markdown_file(filename)
        else:
            raise ValueError("unknown training file format : {} for "
                             "file {}".format(fformat, filename))


    @staticmethod
    def _guess_format(filename):
        # type: (Text) -> Text
        """Applies heuristics to guess the data format of a file."""
        guess = UNK_FILE_FORMAT
        content = utils.read_file(filename)
        try:
            json_content = json.loads(content)
        except ValueError:
            if "## intent:" in content:
                guess = MARKDOWN_FILE_FORMAT
        else:
            if "data" in json_content and type(json_content.get("data")) is list:
                guess = WIT_FILE_FORMAT
            elif "luis_schema_version" in json_content:
                guess = LUIS_FILE_FORMAT
            elif "supportedLanguages" in json_content:
                guess = DIALOGFLOW_FILE_FORMAT
            elif "rasa_nlu_data" in json_content:
                guess = RASA_FILE_FORMAT

        return guess