# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import pytest
from rasa_nlu.training_data import TrainingData

from rasa_nlu.converters import load_data

from rasa_nlu.train import create_persistor

from rasa_nlu.model import Trainer

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
    assert loaded.parse("hello") is not None
    assert loaded.parse("Hello today is Monday, again!") is not None


@slowtest
def test_train_model_noents(component_builder):
    _config = utilities.base_test_conf("all_components")
    _config['data'] = "./data/examples/rasa/demo-rasa-noents.json"
    (trained, persisted_path) = utilities.run_train(_config, component_builder)
    assert trained.pipeline
    loaded = utilities.load_interpreter_for_model(_config, persisted_path, component_builder)
    assert loaded.pipeline
    assert loaded.parse("hello") is not None
    assert loaded.parse("Hello today is Monday, again!") is not None


@slowtest
def test_train_model_multithread(component_builder):
    _config = utilities.base_test_conf("all_components")
    _config['num_threads'] = 2
    (trained, persisted_path) = utilities.run_train(_config, component_builder)
    assert trained.pipeline
    loaded = utilities.load_interpreter_for_model(_config, persisted_path, component_builder)
    assert loaded.pipeline
    assert loaded.parse("hello") is not None
    assert loaded.parse("Hello today is Monday, again!") is not None


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


def test_handles_pipeline_with_non_existing_component(component_builder):
    _config = utilities.base_test_conf("spacy_sklearn")
    _config['pipeline'].append("my_made_up_component")
    with pytest.raises(Exception) as execinfo:
        utilities.run_train(_config, component_builder)
    assert "Failed to find component" in str(execinfo.value)


def test_load_and_persist_without_train(component_builder):
    _config = utilities.base_test_conf("all_components")
    trainer = Trainer(_config, component_builder)
    persistor = create_persistor(_config)
    persisted_path = trainer.persist(_config['path'], persistor, model_name=_config['name'])
    loaded = utilities.load_interpreter_for_model(_config, persisted_path, component_builder)
    assert loaded.pipeline
    assert loaded.parse("hello") is not None
    assert loaded.parse("Hello today is Monday, again!") is not None


def test_train_with_empty_data(component_builder):
    _config = utilities.base_test_conf("all_components")
    trainer = Trainer(_config, component_builder)
    trainer.train(TrainingData())
    persistor = create_persistor(_config)
    persisted_path = trainer.persist(_config['path'], persistor, model_name=_config['name'])
    loaded = utilities.load_interpreter_for_model(_config, persisted_path, component_builder)
    assert loaded.pipeline
    assert loaded.parse("hello") is not None
    assert loaded.parse("Hello today is Monday, again!") is not None
