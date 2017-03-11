import pytest

import utilities
from rasa_nlu.pipeline import registered_pipelines


@pytest.mark.parametrize("pipeline_name", registered_pipelines.keys())
def test_samples(pipeline_name, interpreter_builder):
    interpreter = utilities.interpreter_for(interpreter_builder, utilities.base_test_conf(pipeline_name))
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
        assert result['intent']['name'] in available_intents, \
            "Wrong intent for sample '{}'".format(text)
        assert result['intent']['confidence'] >= 0, \
            "Low confidence for sample '{}'".format(text)

        # This ensures the model doesn't detect entities that are not present
        # Models on our test data set are not stable enough to require the entities to be found
        for entity in result['entities']:
            assert entity in gold['entities'], \
                "Wrong entities for sample '{}'".format(text)
