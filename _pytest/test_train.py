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
def test_train_model(pipeline_template, component_builder):
    _config = utilities.base_test_conf(pipeline_template)
    (trained, persisted_path) = utilities.run_train(_config, component_builder)
    assert trained.pipeline
    loaded = utilities.load_interpreter_for_model(_config, persisted_path, component_builder)
    assert loaded.pipeline


@slowtest
@pytest.mark.parametrize("pipeline_template", list(registry.registered_pipeline_templates.keys()))
def test_train_model_noents(pipeline_template, component_builder):
    _config = utilities.base_test_conf(pipeline_template)
    _config['data'] = "./data/examples/rasa/demo-rasa-noents.json"
    (trained, persisted_path) = utilities.run_train(_config, component_builder)
    assert trained.pipeline
    loaded = utilities.load_interpreter_for_model(_config, persisted_path, component_builder)
    assert loaded.pipeline
    assert loaded.parse("hello") is not None


@slowtest
@pytest.mark.parametrize("pipeline_template", list(registry.registered_pipeline_templates.keys()))
def test_train_model_multithread(pipeline_template, component_builder):
    _config = utilities.base_test_conf(pipeline_template)
    _config['num_threads'] = 2
    (trained, persisted_path) = utilities.run_train(_config, component_builder)
    assert trained.pipeline
    loaded = utilities.load_interpreter_for_model(_config, persisted_path, component_builder)
    assert loaded.pipeline


def test_train_model_empty_pipeline(component_builder):
    _config = utilities.base_test_conf(pipeline_template=None)   # Should return an empty pipeline
    with pytest.raises(ValueError):
        utilities.run_train(_config, component_builder)


def test_train_named_model(component_builder):
    _config = utilities.base_test_conf("keyword")
    _config['name'] = "my_keyword_model"
    (trained, persisted_path) = utilities.run_train(_config, component_builder)
    assert trained.pipeline
    assert persisted_path.strip("/\\").endswith("my_keyword_model")    # should be saved in a dir named after model
