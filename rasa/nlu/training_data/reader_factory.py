from rasa.nlu.training_data.formats import SupportedFormats
from typing import Optional, Text
from rasa.nlu.training_data.formats.readerwriter import TrainingDataReader


class ReaderFactory:
    @staticmethod
    def get_reader(file_format: Text) -> Optional["TrainingDataReader"]:

        """Generates the appropriate reader class based on the file format."""
        from rasa.nlu.training_data.formats import (
            MarkdownReader,
            WitReader,
            LuisReader,
            RasaReader,
            DialogFlowReader,
        )

        reader = None
        if file_format == SupportedFormats.LUIS:
            reader = LuisReader()
        elif file_format == SupportedFormats.WIT:
            reader = WitReader()
        elif file_format in SupportedFormats.DIALOGFLOW_RELEVANT:
            reader = DialogFlowReader()
        elif file_format == SupportedFormats.RASA:
            reader = RasaReader()
        elif file_format == SupportedFormats.MARKDOWN:
            reader = MarkdownReader()
        return reader
