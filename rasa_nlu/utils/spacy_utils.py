from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from typing import Optional

from rasa_nlu.components import Component
from rasa_nlu.model import Metadata


class SpacyNLP(Component):
    name = "init_spacy"

    context_provides = {
        "pipeline_init": ["spacy_nlp"],
        "process": ["spacy_doc"],
    }

    def __init__(self, nlp=None):
        # type: (Optional[Language]) -> None
        from spacy.language import Language

        self.nlp = nlp

    @classmethod
    def cache_key(cls, model_metadata):
        # type: (Metadata) -> str

        return cls.name + "-" + model_metadata.language

    def pipeline_init(self, language, fine_tune_spacy_ner):
        # type: (str, Optional[bool]) -> dict
        import spacy

        # If fine tuning is disabled, we do not need to load the spacy entity model
        if self.nlp is None:
            if fine_tune_spacy_ner:
                self.nlp = spacy.load(language, parser=False)
            else:
                self.nlp = spacy.load(language, parser=False, entity=False)
        SpacyNLP.ensure_proper_language_model(self.nlp)
        return {"spacy_nlp": self.nlp}

    def process(self, text):
        # type: (str) -> dict

        return {
            "spacy_doc": self.nlp(text, entity=False)  # need to set entity=false, otherwise it interferes with our NER
        }

    @staticmethod
    def ensure_proper_language_model(nlp):
        # type: (Optional[Language]) -> None
        """Checks if the spacy language model is properly loaded. Raises an exception if the model is invalid."""
        from spacy.language import Language

        if nlp is None:
            raise Exception("Failed to load spacy language model. Loading the model returned 'None'.")
        if nlp.path is None:
            # Spacy sets the path to `None` if it did not load the model from disk.
            # In this case `nlp` is an unusable stub.
            raise Exception("Failed to load spacy language model for lang '{}'. ".format(nlp.lang) +
                            "Make sure you have downloaded the correct model (https://spacy.io/docs/usage/).")
