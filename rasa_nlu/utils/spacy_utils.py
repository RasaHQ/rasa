from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

import typing
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Text

from rasa_nlu.components import Component
from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.training_data import Message
from rasa_nlu.training_data import TrainingData

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from spacy.language import Language
    from rasa_nlu.model import Metadata


class SpacyNLP(Component):
    name = "nlp_spacy"

    provides = ["spacy_doc", "spacy_nlp"]

    def __init__(self, nlp, language, spacy_model_name):
        # type: (Language, Text, Text) -> None

        self.nlp = nlp
        self.language = language
        self.spacy_model_name = spacy_model_name

    @classmethod
    def required_packages(cls):
        # type: () -> List[Text]
        return ["spacy"]

    @classmethod
    def create(cls, config):
        # type: (RasaNLUConfig) -> SpacyNLP
        import spacy
        spacy_model_name = config["spacy_model_name"]
        if spacy_model_name is None:
            spacy_model_name = config["language"]
        logger.info("Trying to load spacy model with name '{}'".format(spacy_model_name))
        nlp = spacy.load(spacy_model_name, parser=False)
        cls.ensure_proper_language_model(nlp)
        return SpacyNLP(nlp, config["language"], spacy_model_name)

    @classmethod
    def cache_key(cls, model_metadata):
        # type: (Metadata) -> Text

        spacy_model_name = model_metadata.metadata.get("spacy_model_name")
        if spacy_model_name is None:
            # Fallback, use the language name, e.g. "en", as the model name if no explicit name is defined
            spacy_model_name = model_metadata.language
        return cls.name + "-" + spacy_model_name

    def provide_context(self):
        # type: () -> Dict[Text, Any]

        return {"spacy_nlp": self.nlp}

    def train(self, training_data, config, **kwargs):
        # type: (TrainingData) -> Dict[Text, Any]

        for example in training_data.training_examples:
            example.set("spacy_doc", self.nlp(example.text))

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None

        message.set("spacy_doc", self.nlp(message.text))

    def persist(self, model_dir):
        # type: (Text) -> Dict[Text, Any]

        return {
            "spacy_model_name": self.spacy_model_name,
            "language": self.language
        }

    @classmethod
    def load(cls, model_dir=None, model_metadata=None, cached_component=None, **kwargs):
        # type: (Text, Metadata, Optional[SpacyNLP], **Any) -> SpacyNLP
        import spacy

        if cached_component:
            return cached_component

        nlp = spacy.load(model_metadata.get("spacy_model_name"), parser=False)
        cls.ensure_proper_language_model(nlp)
        return SpacyNLP(nlp, model_metadata.get("language"), model_metadata.get("spacy_model_name"))

    @staticmethod
    def ensure_proper_language_model(nlp):
        # type: (Optional[Language]) -> None
        """Checks if the spacy language model is properly loaded. Raises an exception if the model is invalid."""

        if nlp is None:
            raise Exception("Failed to load spacy language model. Loading the model returned 'None'.")
        if nlp.path is None:
            # Spacy sets the path to `None` if it did not load the model from disk.
            # In this case `nlp` is an unusable stub.
            raise Exception("Failed to load spacy language model for lang '{}'. ".format(nlp.lang) +
                            "Make sure you have downloaded the correct model (https://spacy.io/docs/usage/).")
