SPACY_BACKEND_NAME = "spacy_sklearn"


def ensure_proper_language_model(nlp):
    """Checks if the spacy language model is properly loaded. Raises an exception if the model is invalid."""

    if nlp is None:
        raise Exception("Failed to load spacy language model. Loading the model returned 'None'.")
    if nlp.path is None:
        # Spacy sets the path to `None` if it did not load the model from disk. In this case `nlp` is an unusable stub.
        raise Exception("Failed to load spacy language model for lang '{}'. ".format(nlp.lang) +
                        "Make sure you have downloaded the correct model (https://spacy.io/docs/usage/).")
