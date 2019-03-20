import rasa_nlu

import pytest

from rasa_nlu import registry, training_data
from rasa_nlu.model import Interpreter
from tests import utilities


@utilities.slowtest
@pytest.mark.parametrize("pipeline_template",
                         list(registry.registered_pipeline_templates.keys()))
def test_interpreter(pipeline_template, component_builder, tmpdir):
    test_data = "data/examples/rasa/demo-rasa.json"
    _conf = utilities.base_test_conf(pipeline_template)
    _conf["data"] = test_data
    td = training_data.load_data(test_data)
    interpreter = utilities.interpreter_for(component_builder,
                                            "data/examples/rasa/demo-rasa.json",
                                            tmpdir.strpath,
                                            _conf)

    texts = ["good bye", "i am looking for an indian spot"]

    for text in texts:
        result = interpreter.parse(text, time=None)
        assert result['text'] == text
        assert (not result['intent']['name'] or
                result['intent']['name'] in td.intents)
        assert result['intent']['confidence'] >= 0
        # Ensure the model doesn't detect entity types that are not present
        # Models on our test data set are not stable enough to
        # require the exact entities to be found
        for entity in result['entities']:
            assert entity['entity'] in td.entities


@pytest.mark.parametrize("metadata",
                         [{"rasa_nlu_version": "0.11.0"},
                          {"rasa_nlu_version": "0.10.2"},
                          {"rasa_nlu_version": "0.12.0a1"},
                          {"rasa_nlu_version": "0.12.2"},
                          {"rasa_nlu_version": "0.12.3"},
                          {"rasa_nlu_version": "0.13.3"},
                          {"rasa_nlu_version": "0.13.4"},
                          {"rasa_nlu_version": "0.13.5"},
                          {"rasa_nlu_version": "0.14.0a1"},
                          {"rasa_nlu_version": "0.14.0"},
                          {"rasa_nlu_version": "0.14.1"},
                          {"rasa_nlu_version": "0.14.2"},
                          {"rasa_nlu_version": "0.14.3"},
                          {"rasa_nlu_version": "0.14.4"},
                          {"rasa_nlu_version": "0.15.0a1"}])
def test_model_not_compatible(metadata):
    with pytest.raises(rasa_nlu.model.UnsupportedModelError):
        Interpreter.ensure_model_compatibility(metadata)


@pytest.mark.parametrize("metadata",
                         [{"rasa_nlu_version": "0.15.0a2"},
                          {"rasa_nlu_version": "0.15.0"}])
def test_model_is_compatible(metadata):
    # should not raise an exception
    assert Interpreter.ensure_model_compatibility(metadata) is None
