import pytest

from rasa.nlu.model import InvalidModelError
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.utils.spacy_utils import SpacyNLP


def test_model_fallback_raises_error():
    with pytest.raises(InvalidModelError):
        SpacyNLP.check_model_fallback(None, "xx", warn=True)


supported_langauges = [
    "zh",
    "da",
    "nl",
    "en",
    "fr",
    "de",
    "el",
    "it",
    "ja",
    "lt",
    "mk",
    "nb",
    "pl",
    "pt",
    "ro",
    "ru",
    "es",
]


@pytest.mark.parametrize("lang", supported_langauges)
def test_model_fallback_raises_warning(lang):
    with pytest.warns(UserWarning):
        SpacyNLP.check_model_fallback(None, lang, warn=True)


def test_model_raises_error_not_exist():
    with pytest.raises(InvalidModelError):
        SpacyNLP.create({"model": "dinosaurhead"}, RasaNLUModelConfig())
