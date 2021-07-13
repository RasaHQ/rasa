import pytest

from rasa.nlu.model import InvalidModelError
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.utils.spacy_utils import SpacyNLP


def test_model_raises_error_not_exist():
    """It should throw a direct error when a bad model is passed."""
    with pytest.raises(InvalidModelError) as err:
        SpacyNLP.create({"model": "dinosaurhead"}, RasaNLUModelConfig())
    assert "Please confirm that dinosaurhead is an available spaCy model" in str(err.value)


def test_model_raises_error_no_model():
    """It should throw a direct error when no model is passed."""
    with pytest.raises(InvalidModelError) as err:
        SpacyNLP.create({}, RasaNLUModelConfig())
    assert "Missing model configuration for `SpacyNLP` in `config.yml`" in str(err.value)


def test_cache_key_raises_error():
    """The cache_key is created before the rest of the model. Error also needs to be raised there."""
    with pytest.raises(InvalidModelError) as err:
        SpacyNLP.cache_key(component_meta={}, model_metadata={})
    assert "Missing model configuration for `SpacyNLP` in `config.yml`" in str(err.value)
