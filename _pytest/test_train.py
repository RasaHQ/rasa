# -*- coding: utf-8 -*-

import pytest

import utilities
from rasa_nlu import registry


@pytest.mark.parametrize("template_name", registry.registered_model_templates.keys())
def test_train_model(template_name, interpreter_builder):
    _config = utilities.base_test_conf(template_name)
    (trained, persisted_path) = utilities.run_train(_config)
    assert trained.pipeline
    loaded = utilities.load_interpreter_for_model(_config, persisted_path, interpreter_builder)
    assert loaded.pipeline


@pytest.mark.parametrize("pipeline_name", registry.registered_model_templates.keys())
def test_train_model_noents(pipeline_name, interpreter_builder):
    _config = utilities.base_test_conf(pipeline_name)
    _config['data'] = "./data/examples/rasa/demo-rasa-noents.json"
    (trained, persisted_path) = utilities.run_train(_config)
    assert trained.pipeline
    loaded = utilities.load_interpreter_for_model(_config, persisted_path, interpreter_builder)
    assert loaded.pipeline


@pytest.mark.parametrize("template_name", registry.registered_model_templates.keys())
def test_train_model_multithread(template_name, interpreter_builder):
    _config = utilities.base_test_conf(template_name)
    _config['num_threads'] = 2
    (trained, persisted_path) = utilities.run_train(_config)
    assert trained.pipeline
    loaded = utilities.load_interpreter_for_model(_config, persisted_path, interpreter_builder)
    assert loaded.pipeline


def test_train_spacy_sklearn_finetune_ner(interpreter_builder):
    _config = utilities.base_test_conf("spacy_sklearn")
    _config['fine_tune_spacy_ner'] = True
    (trained, persisted_path) = utilities.run_train(_config)
    assert trained.pipeline
    loaded = utilities.load_interpreter_for_model(_config, persisted_path, interpreter_builder)
    result = loaded.parse(u"I am living in New York now.")
    entities = result['entities']
    # Although the model is trained on restaurant entities, we can use the entities (`GPE`, `DATE`)
    # from spacy since we are fine tuning. This should even be the case if the rasa-entity training data changes!
    assert {u'start': 15, u'end': 23, u'value': u'New York', u'entity': u'GPE'} in entities
    assert {u'start': 24, u'end': 27, u'value': u'now', u'entity': u'DATE'} in entities
