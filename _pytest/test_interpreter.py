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
def test_samples(backend_name, spacy_nlp_en):
    interpreter = utilities.interpreter_for(spacy_nlp_en, utilities.base_test_conf(backend_name))
    available_intents = ["greet", "restaurant_search", "affirm", "goodbye"]
    samples = [
        (
            u"good bye",
            {
                'intent': 'goodbye',
                'entities': []
            }
        ),
        (
            u"i am looking for an indian spot",
            {
                'intent': 'restaurant_search',
                'entities': [{"start": 20, "end": 26, "value": "indian", "entity": "cuisine"}]
            }
        )
    ]

    for text, gold in samples:
        result = interpreter.parse(text)
        assert result['text'] == text, \
            "Wrong text for sample '{}'".format(text)
        assert result['intent'] in available_intents, \
            "Wrong intent for sample '{}'".format(text)
        assert result['confidence'] >= 0, \
            "Low confidence for sample '{}'".format(text)

        # This ensures the model doesn't detect entities that are not present
        # Models on our test data set are not stable enough to require the entities to be found
        for entity in result['entities']:
            assert entity in gold['entities'], \
                "Wrong entities for sample '{}'".format(text)
