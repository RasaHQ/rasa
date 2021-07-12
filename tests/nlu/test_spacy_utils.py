import pytest

from rasa.nlu.model import InvalidModelError
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.utils.spacy_utils import SpacyNLP


@pytest.mark.parametrize(
    "model_name,msg",
    [
        (
            "dinosaurhead",
            "Please confirm that dinosaurhead is an available spaCy model",
        ),
        (None, "Missing model configuration for `SpacyNLP` in `config.yml`"),
    ],
)
def test_model_raises_error_not_exist(model_name, msg):
    """It should throw a direct error when a bad model is passed."""
    with pytest.raises(InvalidModelError) as err:
        SpacyNLP.create({"model": model_name}, RasaNLUModelConfig())
    assert msg in str(err.value)
