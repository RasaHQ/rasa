from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import pytest

from rasa_nlu import registry, training_data
from tests import utilities


@utilities.slowtest
@pytest.mark.parametrize("pipeline_template", list(registry.registered_pipeline_templates.keys()))
def test_interpreter(pipeline_template, component_builder):
    test_data = "data/examples/rasa/demo-rasa.json"
    _conf = utilities.base_test_conf(pipeline_template)
    _conf["data"] = test_data
    td = training_data.load_data(test_data)
    interpreter = utilities.interpreter_for(component_builder, _conf)

    texts = ["good bye", "i am looking for an indian spot"]

    for text in texts:
        result = interpreter.parse(text, time=None)
        assert result['text'] == text
        assert not result['intent']['name'] or result['intent']['name'] in td.intents
        assert result['intent']['confidence'] >= 0
        # Ensure the model doesn't detect entity types that are not present
        # Models on our test data set are not stable enough to require the exact entities to be found
        for entity in result['entities']:
            assert entity['entity'] in td.entities
