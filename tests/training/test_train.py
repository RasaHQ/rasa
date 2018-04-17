# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import pytest
from rasa_nlu.config import RasaNLUModelConfig

from tests.conftest import DEFAULT_DATA_PATH
from rasa_nlu import registry, train
from rasa_nlu.model import Trainer, Interpreter
from rasa_nlu.train import create_persistor
from rasa_nlu.training_data import TrainingData
from tests import utilities


def as_pipeline(*components):
    return [{"name": c} for c in components]


def pipelines_for_tests():
    # these templates really are just for testing
    # every component should be in here so train-persist-load-use cycle can be
    # tested they still need to be in a useful order - hence we can not simply
    # generate this automatically.

    # first is language followed by list of components
    return [("en", as_pipeline("nlp_spacy",
                               "nlp_mitie",
                               "tokenizer_whitespace",
                               "tokenizer_mitie",
                               "tokenizer_spacy",
                               "intent_featurizer_mitie",
                               "intent_featurizer_spacy",
                               "intent_featurizer_ngrams",
                               "intent_entity_featurizer_regex",
                               "intent_featurizer_count_vectors",
                               "ner_mitie",
                               "ner_crf",
                               "ner_spacy",
                               "ner_duckling",
                               "ner_duckling_http",
                               "ner_synonyms",
                               "intent_classifier_keyword",
                               "intent_classifier_sklearn",
                               "intent_classifier_mitie",
                               "intent_classifier_tensorflow_embedding"
                               )),
            ("zh", as_pipeline("nlp_mitie",
                               "tokenizer_jieba",
                               "intent_featurizer_mitie",
                               "ner_mitie",
                               "intent_classifier_sklearn",
                               )),
            ]


def test_all_components_are_in_at_least_one_test_pipeline():
    """There is a template that includes all components to
    test the train-persist-load-use cycle. Ensures that
    really all Components are in there."""

    all_components = [c["name"] for _, p in pipelines_for_tests() for c in p]
    for cls in registry.component_classes:
        assert cls.name in all_components, \
            "`all_components` template is missing component."


@utilities.slowtest
@pytest.mark.parametrize("pipeline_template",
                         list(registry.registered_pipeline_templates.keys()))
def test_train_model(pipeline_template, component_builder, tmpdir):
    _config = utilities.base_test_conf(pipeline_template)
    (trained, _, persisted_path) = train.do_train(
            _config,
            path=tmpdir.strpath,
            data=DEFAULT_DATA_PATH,
            component_builder=component_builder)
    assert trained.pipeline
    loaded = Interpreter.load(persisted_path, component_builder)
    assert loaded.pipeline
    assert loaded.parse("hello") is not None
    assert loaded.parse("Hello today is Monday, again!") is not None


@utilities.slowtest
@pytest.mark.parametrize("language, pipeline", pipelines_for_tests())
def test_train_model_on_test_pipelines(language, pipeline,
                                       component_builder, tmpdir):
    _config = RasaNLUModelConfig({"pipeline": pipeline, "language": language})
    (trained, _, persisted_path) = train.do_train(
            _config,
            path=tmpdir.strpath,
            data=DEFAULT_DATA_PATH,
            component_builder=component_builder)
    assert trained.pipeline
    loaded = Interpreter.load(persisted_path, component_builder)
    assert loaded.pipeline
    assert loaded.parse("hello") is not None
    assert loaded.parse("Hello today is Monday, again!") is not None


@utilities.slowtest
@pytest.mark.parametrize("language, pipeline", pipelines_for_tests())
def test_train_model_noents(language, pipeline, component_builder, tmpdir):
    _config = RasaNLUModelConfig({"pipeline": pipeline, "language": language})
    (trained, _, persisted_path) = train.do_train(
            _config,
            path=tmpdir.strpath,
            data="./data/test/demo-rasa-noents.json",
            component_builder=component_builder)
    assert trained.pipeline
    loaded = Interpreter.load(persisted_path, component_builder)
    assert loaded.pipeline
    assert loaded.parse("hello") is not None
    assert loaded.parse("Hello today is Monday, again!") is not None


@utilities.slowtest
@pytest.mark.parametrize("language, pipeline", pipelines_for_tests())
def test_train_model_multithread(language, pipeline, component_builder, tmpdir):
    _config = RasaNLUModelConfig({"pipeline": pipeline, "language": language})
    (trained, _, persisted_path) = train.do_train(
            _config,
            path=tmpdir.strpath,
            data=DEFAULT_DATA_PATH,
            component_builder=component_builder,
            num_threads=2)
    assert trained.pipeline
    loaded = Interpreter.load(persisted_path, component_builder)
    assert loaded.pipeline
    assert loaded.parse("hello") is not None
    assert loaded.parse("Hello today is Monday, again!") is not None


def test_train_model_empty_pipeline(component_builder):
    # Should return an empty pipeline
    _config = utilities.base_test_conf(pipeline_template=None)
    with pytest.raises(ValueError):
        train.do_train(
                _config,
                data=DEFAULT_DATA_PATH,
                component_builder=component_builder)


def test_train_named_model(component_builder, tmpdir):
    _config = utilities.base_test_conf("keyword")
    (trained, _, persisted_path) = train.do_train(
            _config,
            path=tmpdir.strpath,
            project="my_keyword_model",
            data=DEFAULT_DATA_PATH,
            component_builder=component_builder)
    assert trained.pipeline
    normalized_path = os.path.dirname(os.path.normpath(persisted_path))
    # should be saved in a dir named after a project
    assert os.path.basename(normalized_path) == "my_keyword_model"


def test_handles_pipeline_with_non_existing_component(component_builder):
    _config = utilities.base_test_conf("spacy_sklearn")
    _config.pipeline.append({"name": "my_made_up_component"})
    with pytest.raises(Exception) as execinfo:
        train.do_train(
                _config,
                data=DEFAULT_DATA_PATH,
                component_builder=component_builder)
    assert "Failed to find component" in str(execinfo.value)


@pytest.mark.parametrize("language, pipeline", pipelines_for_tests())
def test_load_and_persist_without_train(language, pipeline,
                                        component_builder, tmpdir):
    _config = RasaNLUModelConfig({"pipeline": pipeline, "language": language})
    trainer = Trainer(_config, component_builder)
    persistor = create_persistor(_config)
    persisted_path = trainer.persist(tmpdir.strpath, persistor,
                                     project_name="my_project")
    loaded = Interpreter.load(persisted_path, component_builder)
    assert loaded.pipeline
    assert loaded.parse("hello") is not None
    assert loaded.parse("Hello today is Monday, again!") is not None


@pytest.mark.parametrize("language, pipeline", pipelines_for_tests())
def test_train_with_empty_data(language, pipeline, component_builder, tmpdir):
    _config = RasaNLUModelConfig({"pipeline": pipeline, "language": language})
    trainer = Trainer(_config, component_builder)
    trainer.train(TrainingData())
    persistor = create_persistor(_config)
    persisted_path = trainer.persist(tmpdir.strpath, persistor,
                                     project_name="my_project")
    loaded = Interpreter.load(persisted_path, component_builder)
    assert loaded.pipeline
    assert loaded.parse("hello") is not None
    assert loaded.parse("Hello today is Monday, again!") is not None
