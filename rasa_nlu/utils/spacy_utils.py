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
from rasa_nlu.model import Metadata


if typing.TYPE_CHECKING:
    from spacy.language import Language


class SpacyNLP(Component):
    name = "nlp_spacy"

    context_provides = {
        "pipeline_init": ["spacy_nlp"],
        "process": ["spacy_doc"],
    }

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
    def create(cls, language, spacy_model_name):
        # type: (Text, Text) -> SpacyNLP
        import spacy

        if spacy_model_name is None:
            spacy_model_name = language
        logging.info("Trying to load spacy model with name '{}'".format(spacy_model_name))
        nlp = spacy.load(spacy_model_name, parser=False)
        spacy_model_name = spacy_model_name
        cls.ensure_proper_language_model(nlp)
        return SpacyNLP(nlp, language, spacy_model_name)

    @classmethod
    def cache_key(cls, model_metadata):
        # type: (Metadata) -> Text

        spacy_model_name = model_metadata.metadata.get("spacy_model_name")
        if spacy_model_name is None:
            # Fallback, use the language name, e.g. "en", as the model name if no explicit name is defined
            spacy_model_name = model_metadata.language
        return cls.name + "-" + spacy_model_name

    def pipeline_init(self):
        # type: () -> Dict[Text, Any]

        return {"spacy_nlp": self.nlp}

    def process(self, text):
        # type: (Text) -> Dict[Text, Any]

        return {
            "spacy_doc": self.nlp(text, entity=False)  # need to set entity=false, otherwise it interferes with our NER
        }

    def persist(self, model_dir):
        # type: (Text) -> Dict[Text, Any]

        return {
            "spacy_model_name": self.spacy_model_name,
            "language": self.language
        }

    @classmethod
    def load(cls, language, spacy_model_name):
        # type: (Text, Text) -> SpacyNLP

        return cls.create(language, spacy_model_name)

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
