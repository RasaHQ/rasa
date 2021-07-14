import pytest

from rasa.nlu.model import InvalidModelError
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.utils.spacy_utils import SpacyNLP


@pytest.mark.parametrize(
    "params,msg",
    [
        (
            {"model": "dinosaurhead"},
            "Please confirm that dinosaurhead is an available spaCy model",
        ),
        ({}, "Missing model configuration for `SpacyNLP` in `config.yml`"),
    ],
)
def test_model_raises_error_not_exist(params, msg):
    """It should throw a direct error when a bad model setting goes in."""
    with pytest.raises(InvalidModelError) as err:
        SpacyNLP.create(params, RasaNLUModelConfig())
    assert msg in str(err.value)


def test_cache_key_raises_error():
    """
    The cache_key is created before the rest of the model.
    Error also needs to be raised there.
    """
    with pytest.raises(InvalidModelError) as err:
        SpacyNLP.cache_key(component_meta={}, model_metadata={})
    msg = "Missing model configuration for `SpacyNLP` in `config.yml`"
    assert msg in str(err.value)
