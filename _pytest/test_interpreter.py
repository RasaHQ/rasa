import pytest

import utilities
from rasa_nlu.utils.mitie import MITIE_BACKEND_NAME
from rasa_nlu.utils.mitie import MITIE_SKLEARN_BACKEND_NAME
from rasa_nlu.utils.spacy import SPACY_BACKEND_NAME


@pytest.mark.parametrize("backend_name", [
    MITIE_BACKEND_NAME,
    MITIE_SKLEARN_BACKEND_NAME,
    SPACY_BACKEND_NAME,
])
def test_samples(backend_name):
    interpreter = utilities.interpreter_for(utilities.base_test_conf(backend_name))
    samples = [
        (
            u"good bye",
            {
                'intent': 'goodbye',
                'entities': [],
                'min_confidence': 0.3
            }
        ),
        (
            u"i am looking for an indian spot",
            {
                'intent': 'restaurant_search',
                'entities': [{"start": 20, "end": 26, "value": "indian", "entity": "cuisine"}],
                'min_confidence': 0.3
            }
        )
    ]

    for text, gold in samples:
        result = interpreter.parse(text)
        assert result['text'] == text, \
            "Wrong text for sample '{}'".format(text)
        assert result['intent'] == gold['intent'], \
            "Wrong intent for sample '{}'".format(text)
        assert result['confidence'] >= gold['min_confidence'], \
            "Low confidence for sample '{}'".format(text)
        assert result['entities'] == gold['entities'], \
            "Wrong entities for sample '{}'".format(text)
