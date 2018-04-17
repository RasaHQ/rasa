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
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.training_data import Message
from rasa_nlu.training_data import TrainingData

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from spacy.language import Language
    from rasa_nlu.model import Metadata


class SpacyNLP(Component):
    name = "nlp_spacy"

    provides = ["spacy_doc", "spacy_nlp"]

    defaults = {
        # name of the language model to load - if it is not set
        # we will be looking for a language model that is named
        # after the language of the model, e.g. `en`
        "model": None,

        # when retrieving word vectors, this will decide if the casing
        # of the word is relevant. E.g. `hello` and `Hello` will
        # retrieve the same vector, if set to `False`. For some
        # applications and models it makes sense to differentiate
        # between these two words, therefore setting this to `True`.
        "case_sensitive": False,
    }

    def __init__(self, component_config=None, nlp=None):
        # type: (Dict[Text, Any], Language) -> None

        self.nlp = nlp
        super(SpacyNLP, self).__init__(component_config)

    @classmethod
    def required_packages(cls):
        # type: () -> List[Text]
        return ["spacy"]

    @classmethod
    def create(cls, cfg):
        # type: (RasaNLUModelConfig) -> SpacyNLP
        import spacy

        component_conf = cfg.for_component(cls.name, cls.defaults)
        spacy_model_name = component_conf.get("model")

        # if no model is specified, we fall back to the language string
        if not spacy_model_name:
            spacy_model_name = cfg.language
            component_conf["model"] = cfg.language

        logger.info("Trying to load spacy model with "
                    "name '{}'".format(spacy_model_name))

        nlp = spacy.load(spacy_model_name, parser=False)
        cls.ensure_proper_language_model(nlp)
        return SpacyNLP(component_conf, nlp)

    @classmethod
    def cache_key(cls, model_metadata):
        # type: (Metadata) -> Text

        component_meta = model_metadata.for_component(cls.name)

        # Fallback, use the language name, e.g. "en",
        # as the model name if no explicit name is defined
        spacy_model_name = component_meta.get("model", model_metadata.language)

        return cls.name + "-" + spacy_model_name

    def provide_context(self):
        # type: () -> Dict[Text, Any]

        return {"spacy_nlp": self.nlp}

    def doc_for_text(self, text):
        if self.component_config.get("case_sensitive"):
            return self.nlp(text)
        else:
            return self.nlp(text.lower())

    def train(self, training_data, config, **kwargs):
        # type: (TrainingData, RasaNLUModelConfig, **Any) -> None

        for example in training_data.training_examples:
            example.set("spacy_doc", self.doc_for_text(example.text))

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None

        message.set("spacy_doc", self.doc_for_text(message.text))

    @classmethod
    def load(cls,
             model_dir=None,
             model_metadata=None,
             cached_component=None,
             **kwargs):
        # type: (Text, Metadata, Optional[SpacyNLP], **Any) -> SpacyNLP
        import spacy

        if cached_component:
            return cached_component

        component_meta = model_metadata.for_component(cls.name)
        model_name = component_meta.get("model")

        nlp = spacy.load(model_name, parser=False)
        cls.ensure_proper_language_model(nlp)
        return cls(component_meta, nlp)

    @staticmethod
    def ensure_proper_language_model(nlp):
        # type: (Optional[Language]) -> None
        """Checks if the spacy language model is properly loaded.

        Raises an exception if the model is invalid."""

        if nlp is None:
            raise Exception("Failed to load spacy language model. "
                            "Loading the model returned 'None'.")
        if nlp.path is None:
            # Spacy sets the path to `None` if
            # it did not load the model from disk.
            # In this case `nlp` is an unusable stub.
            raise Exception("Failed to load spacy language model for "
                            "lang '{}'. Make sure you have downloaded the "
                            "correct model (https://spacy.io/docs/usage/)."
                            "".format(nlp.lang))
