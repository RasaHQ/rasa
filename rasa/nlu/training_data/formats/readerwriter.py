import json

import rasa.utils.io
import typing
from rasa.nlu import utils
from typing import NoReturn, Text, Dict, Any

if typing.TYPE_CHECKING:
    from rasa.nlu.training_data import TrainingData


class TrainingDataReader:
    def read(self, filename: Text, **kwargs: Any) -> "TrainingData":
        """Reads TrainingData from a file."""
        return self.reads(rasa.utils.io.read_file(filename), **kwargs)

    def reads(self, s: Text, **kwargs: Any) -> "TrainingData":
        """Reads TrainingData from a string."""
        raise NotImplementedError


class TrainingDataWriter:
    def dump(self, filename: Text, training_data) -> None:
        """Writes a TrainingData object in markdown format to a file."""
        s = self.dumps(training_data)
        utils.write_to_file(filename, s)

    def dumps(self, training_data: "TrainingData") -> Text:
        """Turns TrainingData into a string."""
        raise NotImplementedError


class JsonTrainingDataReader(TrainingDataReader):
    def reads(self, s: Text, **kwargs: Any) -> "TrainingData":
        """Transforms string into json object and passes it on."""
        js = json.loads(s)
        return self.read_from_json(js, **kwargs)

    def read_from_json(self, js: Dict[Text, Any], **kwargs: Any) -> "TrainingData":
        """Reads TrainingData from a json object."""
        raise NotImplementedError
