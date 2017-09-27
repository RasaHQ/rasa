from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import pytest

from tests import utilities
from rasa_nlu import registry


@utilities.slowtest
@pytest.mark.parametrize("pipeline_template", list(registry.registered_pipeline_templates.keys()))
def test_samples(pipeline_template, component_builder):
    _conf = utilities.base_test_conf(pipeline_template)
    _conf["data"] = "./data/examples/rasa/demo-rasa.json"

    interpreter = utilities.interpreter_for(component_builder, _conf)
    available_intents = ["greet", "restaurant_search", "affirm", "goodbye", "None"]
    samples = [
        (
            "good bye",
            {
                'intent': 'goodbye',
                'entities': []
            }
        ),
        (
            "i am looking for an indian spot",
            {
                'intent': 'restaurant_search',
                'entities': [{"start": 20, "end": 26, "value": "indian", "entity": "cuisine"}]
            }
        )
    ]

    for text, gold in samples:
        result = interpreter.parse(text, time=None)
        assert result['text'] == text, \
            "Wrong text for sample '{}'".format(text)
        assert result['intent']['name'] in available_intents, \
            "Wrong intent for sample '{}'".format(text)
        assert result['intent']['confidence'] >= 0, \
            "Low confidence for sample '{}'".format(text)

        # This ensures the model doesn't detect entities that are not present
        # Models on our test data set are not stable enough to require the entities to be found
        for entity in result['entities']:
            del entity["extractor"]
            assert entity in gold['entities'], \
                "Wrong entities for sample '{}'".format(text)
