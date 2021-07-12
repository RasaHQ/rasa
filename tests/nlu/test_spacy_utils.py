import pytest

from rasa.nlu.model import InvalidModelError
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.utils.spacy_utils import SpacyNLP


def test_model_raises_error_not_exist():
    """It should throw a direct error when a model doesn't exist."""
    with pytest.raises(InvalidModelError):
        SpacyNLP.create({"model": "dinosaurhead"}, RasaNLUModelConfig())
