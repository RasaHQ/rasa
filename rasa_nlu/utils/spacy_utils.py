from rasa_nlu.components import Component

SPACY_BACKEND_NAME = "spacy_sklearn"


class SpacyNLP(Component):
    name = "init_spacy"

    context_provides = ["spacy_nlp"]

    def __init__(self, nlp=None):
        """
        :type nlp: spacy.language.Language or None
        """
        self.nlp = nlp

    @classmethod
    def cache_key(cls, model_metadata):
        return cls.name + "-" + model_metadata.language

    def pipeline_init(self, language, fine_tune_spacy_ner):
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
        return {
            "spacy_doc": self.nlp(text)
        }

    @staticmethod
    def ensure_proper_language_model(nlp):
        """Checks if the spacy language model is properly loaded. Raises an exception if the model is invalid.
        :type nlp: Language or None
        """

        if nlp is None:
            raise Exception("Failed to load spacy language model. Loading the model returned 'None'.")
        if nlp.path is None:
            # Spacy sets the path to `None` if it did not load the model from disk.
            # In this case `nlp` is an unusable stub.
            raise Exception("Failed to load spacy language model for lang '{}'. ".format(nlp.lang) +
                            "Make sure you have downloaded the correct model (https://spacy.io/docs/usage/).")
