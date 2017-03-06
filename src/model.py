import json
import os


class InvalidModelError(Exception):
    """Raised when a model failed to load.

    Attributes:
        message -- explanation of why the model is invalid
    """

    def __init__(self, message):
        self.message = message


class Model(object):
    def __init__(self, metadata, model_dir, interpreter=None, nlp=None, featurizer=None):
        self.metadata = metadata
        self.model_dir = model_dir
        self.interpreter = interpreter
        self.featurizer = featurizer
        self.nlp = nlp


class Metadata(object):
    @staticmethod
    def load(model_dir):
        with open(os.path.join(model_dir, 'metadata.json'), 'rb') as f:
            data = json.loads(f.read().decode('utf-8'))
        return Metadata(data, model_dir)

    def __init__(self, metadata, model_dir):
        self.metadata = metadata
        self.model_dir = model_dir

    def __prepend_path(self, prop):
        if self.metadata.get(prop) is not None:
            return os.path.normpath(os.path.join(self.model_dir, self.metadata[prop]))
        else:
            return None

    @property
    def feature_extractor_path(self):
        return self.__prepend_path("feature_extractor")

    @property
    def intent_classifier_path(self):
        return self.__prepend_path("intent_classifier")

    @property
    def entity_extractor_path(self):
        return self.__prepend_path("entity_extractor")

    @property
    def entity_synonyms_path(self):
        return self.__prepend_path("entity_synonyms")

    def language_name(self):
        return self.metadata.get('language_name')

    def backend_name(self):
        return self.metadata.get('backend')

    def model_group(self):
        """Groups models by backend and language name."""
        if self.language_name() is not None and self.backend_name() is not None:
            return self.language_name() + "@" + self.backend_name()
        return None
