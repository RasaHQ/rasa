import json

import rasa.utils.io
from rasa.nlu import utils


class TrainingDataReader(object):
    def read(self, filename, **kwargs):
        """Reads TrainingData from a file."""
        return self.reads(rasa.utils.io.read_file(filename), **kwargs)

    def reads(self, s, **kwargs):
        """Reads TrainingData from a string."""
        raise NotImplementedError


class TrainingDataWriter(object):
    def dump(self, filename, training_data):
        """Writes a TrainingData object in markdown format to a file."""
        s = self.dumps(training_data)
        utils.write_to_file(filename, s)

    def dumps(self, training_data):
        """Turns TrainingData into a string."""
        raise NotImplementedError


class JsonTrainingDataReader(TrainingDataReader):
    def reads(self, s, **kwargs):
        """Transforms string into json object and passes it on."""
        js = json.loads(s)
        return self.read_from_json(js, **kwargs)

    def read_from_json(self, js, **kwargs):
        """Reads TrainingData from a json object."""
        raise NotImplementedError
