# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import pytest

import utilities
from utilities import slowtest
from rasa_nlu import registry


@slowtest
@pytest.mark.parametrize("pipeline_template", list(registry.registered_pipeline_templates.keys()))
def test_train_model(pipeline_template, interpreter_builder):
    _config = utilities.base_test_conf(pipeline_template)
    (trained, persisted_path) = utilities.run_train(_config)
    assert trained.pipeline
    loaded = utilities.load_interpreter_for_model(_config, persisted_path, interpreter_builder)
    assert loaded.pipeline


@slowtest
@pytest.mark.parametrize("pipeline_template", list(registry.registered_pipeline_templates.keys()))
def test_train_model_noents(pipeline_template, interpreter_builder):
    _config = utilities.base_test_conf(pipeline_template)
    _config['data'] = "./data/examples/rasa/demo-rasa-noents.json"
    (trained, persisted_path) = utilities.run_train(_config)
    assert trained.pipeline
    loaded = utilities.load_interpreter_for_model(_config, persisted_path, interpreter_builder)
    assert loaded.pipeline


@slowtest
@pytest.mark.parametrize("pipeline_template", list(registry.registered_pipeline_templates.keys()))
def test_train_model_multithread(pipeline_template, interpreter_builder):
    _config = utilities.base_test_conf(pipeline_template)
    _config['num_threads'] = 2
    (trained, persisted_path) = utilities.run_train(_config)
    assert trained.pipeline
    loaded = utilities.load_interpreter_for_model(_config, persisted_path, interpreter_builder)
    assert loaded.pipeline


@slowtest
def test_train_spacy_sklearn_finetune_ner(interpreter_builder):
    _config = utilities.base_test_conf("spacy_sklearn")
    _config['fine_tune_spacy_ner'] = True
    (trained, persisted_path) = utilities.run_train(_config)
    assert trained.pipeline
    loaded = utilities.load_interpreter_for_model(_config, persisted_path, interpreter_builder)
    result = loaded.parse(u"I am living in New York City now.")
    entities = result['entities']
    # Although the model is trained on restaurant entities, we can use the entities (`GPE`, `DATE`)
    # from spacy since we are fine tuning. This should even be the case if the rasa-entity training data changes!
    assert {u'start': 15, u'end': 28, u'value': u'New York City', u'entity': u'GPE'} in entities
