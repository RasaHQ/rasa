import logging
import typing
from typing import Any, Dict, List, Optional, Text

from rasa.nlu.components import Component
from rasa.nlu.config import RasaNLUModelConfig, override_defaults
from rasa.nlu.training_data import Message, TrainingData
from rasa.nlu.model import InvalidModelError

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from spacy.language import Language
    from spacy.tokens.doc import Doc  # pytype: disable=import-error
    from rasa.nlu.model import Metadata


class SpacyNLP(Component):
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

    def __init__(
        self, component_config: Dict[Text, Any] = None, nlp: "Language" = None
    ) -> None:

        self.nlp = nlp
        super(SpacyNLP, self).__init__(component_config)

    @staticmethod
    def load_model(spacy_model_name: Text) -> "Language":
        """Try loading the model, catching the OSError if missing."""
        import spacy

        try:
            return spacy.load(spacy_model_name, disable=["parser"])
        except OSError:
            raise InvalidModelError(
                "Model '{}' is not a linked spaCy model.  "
                "Please download and/or link a spaCy model, "
                "e.g. by running:\npython -m spacy download "
                "en_core_web_md\npython -m spacy link "
                "en_core_web_md en".format(spacy_model_name)
            )

    @classmethod
    def required_packages(cls) -> List[Text]:
        return ["spacy"]

    @classmethod
    def create(
        cls, component_config: Dict[Text, Any], config: RasaNLUModelConfig
    ) -> "SpacyNLP":

        component_config = override_defaults(cls.defaults, component_config)

        spacy_model_name = component_config.get("model")

        # if no model is specified, we fall back to the language string
        if not spacy_model_name:
            spacy_model_name = config.language
            component_config["model"] = config.language

        logger.info(
            "Trying to load spacy model with name '{}'".format(spacy_model_name)
        )

        nlp = cls.load_model(spacy_model_name)

        cls.ensure_proper_language_model(nlp)
        return cls(component_config, nlp)

    @classmethod
    def cache_key(
        cls, component_meta: Dict[Text, Any], model_metadata: "Metadata"
    ) -> Optional[Text]:

        # Fallback, use the language name, e.g. "en",
        # as the model name if no explicit name is defined
        spacy_model_name = component_meta.get("model", model_metadata.language)

        return cls.name + "-" + spacy_model_name

    def provide_context(self) -> Dict[Text, Any]:
        return {"spacy_nlp": self.nlp}

    def doc_for_text(self, text: Text) -> "Doc":
        if self.component_config.get("case_sensitive"):
            return self.nlp(text)
        else:
            return self.nlp(text.lower())

    def train(
        self, training_data: TrainingData, config: RasaNLUModelConfig, **kwargs: Any
    ) -> None:

        for example in training_data.training_examples:
            example.set("spacy_doc", self.doc_for_text(example.text))

    def process(self, message: Message, **kwargs: Any) -> None:

        message.set("spacy_doc", self.doc_for_text(message.text))

    @classmethod
    def load(
        cls,
        meta: Dict[Text, Any],
        model_dir: Text = None,
        model_metadata: "Metadata" = None,
        cached_component: Optional["SpacyNLP"] = None,
        **kwargs: Any
    ) -> "SpacyNLP":

        if cached_component:
            return cached_component

        model_name = meta.get("model")

        nlp = cls.load_model(model_name)
        cls.ensure_proper_language_model(nlp)
        return cls(meta, nlp)

    @staticmethod
    def ensure_proper_language_model(nlp: Optional["Language"]) -> None:
        """Checks if the spacy language model is properly loaded.

        Raises an exception if the model is invalid."""

        if nlp is None:
            raise Exception(
                "Failed to load spacy language model. "
                "Loading the model returned 'None'."
            )
        if nlp.path is None:
            # Spacy sets the path to `None` if
            # it did not load the model from disk.
            # In this case `nlp` is an unusable stub.
            raise Exception(
                "Failed to load spacy language model for "
                "lang '{}'. Make sure you have downloaded the "
                "correct model (https://spacy.io/docs/usage/)."
                "".format(nlp.lang)
            )
