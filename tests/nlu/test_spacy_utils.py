import pytest

from rasa.nlu.model import InvalidModelError
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.utils.spacy_utils import SpacyNLP


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
def test_model_fallback_raises_warning(lang: str):
    """Make sure we raise a warning but we will perform a fallback."""
    with pytest.warns(FutureWarning):
        SpacyNLP._check_model_fallback(None, lang, warn=True)


def test_model_raises_error_not_exist():
    """It should throw a direct error when a model doesn't exist."""
    with pytest.raises(InvalidModelError):
        SpacyNLP.create({"model": "dinosaurhead"}, RasaNLUModelConfig())
