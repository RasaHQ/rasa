# -*- coding: utf-8 -*-

import os
import pytest

from rasa.nlu import registry, train
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.model import Interpreter, Trainer
from rasa.nlu.train import create_persistor
from rasa.nlu.training_data import TrainingData
from tests.nlu import utilities
from tests.nlu.conftest import DEFAULT_DATA_PATH


def as_pipeline(*components):
    return [{"name": c} for c in components]


def pipelines_for_tests():
    # these templates really are just for testing
    # every component should be in here so train-persist-load-use cycle can be
    # tested they still need to be in a useful order - hence we can not simply
    # generate this automatically.

    # first is language followed by list of components
    return [
        (
            "en",
            as_pipeline(
                "SpacyNLP",
                "MitieNLP",
                "WhitespaceTokenizer",
                "MitieTokenizer",
                "SpacyTokenizer",
                "MitieFeaturizer",
                "SpacyFeaturizer",
                "NGramFeaturizer",
                "RegexFeaturizer",
                "CountVectorsFeaturizer",
                "MitieEntityExtractor",
                "CRFEntityExtractor",
                "SpacyEntityExtractor",
                "DucklingHTTPExtractor",
                "EntitySynonymMapper",
                "KeywordIntentClassifier",
                "SklearnIntentClassifier",
                "MitieIntentClassifier",
                "EmbeddingIntentClassifier",
            ),
        ),
        (
            "zh",
            as_pipeline(
                "MitieNLP",
                "JiebaTokenizer",
                "MitieFeaturizer",
                "MitieEntityExtractor",
                "SklearnIntentClassifier",
            ),
        ),
    ]


def test_all_components_are_in_at_least_one_test_pipeline():
    """There is a template that includes all components to
    test the train-persist-load-use cycle. Ensures that
    really all Components are in there."""

    all_components = [c["name"] for _, p in pipelines_for_tests() for c in p]
    for cls in registry.component_classes:
        assert (
            cls.name in all_components
        ), "`all_components` template is missing component."


@utilities.slowtest
@pytest.mark.parametrize(
    "pipeline_template", list(registry.registered_pipeline_templates.keys())
)
def test_train_model(pipeline_template, component_builder, tmpdir):
    _config = utilities.base_test_conf(pipeline_template)
    (trained, _, persisted_path) = train(
        _config,
        path=tmpdir.strpath,
        data=DEFAULT_DATA_PATH,
        component_builder=component_builder,
    )
    assert trained.pipeline
    loaded = Interpreter.load(persisted_path, component_builder)
    assert loaded.pipeline
    assert loaded.parse("hello") is not None
    assert loaded.parse("Hello today is Monday, again!") is not None


@utilities.slowtest
def test_random_seed(component_builder, tmpdir):
    """test if train result is the same for two runs of tf embedding"""

    _config = utilities.base_test_conf("supervised_embeddings")
    # set fixed random seed to 1
    _config.set_component_attr(5, random_seed=1)
    # first run
    (trained_a, _, persisted_path_a) = train(
        _config,
        path=tmpdir.strpath + "_a",
        data=DEFAULT_DATA_PATH,
        component_builder=component_builder,
    )
    # second run
    (trained_b, _, persisted_path_b) = train(
        _config,
        path=tmpdir.strpath + "_b",
        data=DEFAULT_DATA_PATH,
        component_builder=component_builder,
    )
    loaded_a = Interpreter.load(persisted_path_a, component_builder)
    loaded_b = Interpreter.load(persisted_path_b, component_builder)
    result_a = loaded_a.parse("hello")["intent"]["confidence"]
    result_b = loaded_b.parse("hello")["intent"]["confidence"]
    assert result_a == result_b


@utilities.slowtest
@pytest.mark.parametrize("language, pipeline", pipelines_for_tests())
def test_train_model_on_test_pipelines(language, pipeline, component_builder, tmpdir):
    _config = RasaNLUModelConfig({"pipeline": pipeline, "language": language})
    (trained, _, persisted_path) = train(
        _config,
        path=tmpdir.strpath,
        data=DEFAULT_DATA_PATH,
        component_builder=component_builder,
    )
    assert trained.pipeline
    loaded = Interpreter.load(persisted_path, component_builder)
    assert loaded.pipeline
    assert loaded.parse("hello") is not None
    assert loaded.parse("Hello today is Monday, again!") is not None


@utilities.slowtest
@pytest.mark.parametrize("language, pipeline", pipelines_for_tests())
def test_train_model_noents(language, pipeline, component_builder, tmpdir):
    _config = RasaNLUModelConfig({"pipeline": pipeline, "language": language})
    (trained, _, persisted_path) = train(
        _config,
        path=tmpdir.strpath,
        data="./data/test/demo-rasa-noents.json",
        component_builder=component_builder,
    )
    assert trained.pipeline
    loaded = Interpreter.load(persisted_path, component_builder)
    assert loaded.pipeline
    assert loaded.parse("hello") is not None
    assert loaded.parse("Hello today is Monday, again!") is not None


def test_train_model_empty_pipeline(component_builder):
    # Should return an empty pipeline
    _config = utilities.base_test_conf(pipeline_template=None)
    with pytest.raises(ValueError):
        train(_config, data=DEFAULT_DATA_PATH, component_builder=component_builder)


def test_train_named_model(component_builder, tmpdir):
    _config = utilities.base_test_conf("keyword")
    (trained, _, persisted_path) = train(
        _config,
        path=tmpdir.strpath,
        data=DEFAULT_DATA_PATH,
        component_builder=component_builder,
    )
    assert trained.pipeline
    normalized_path = os.path.dirname(os.path.normpath(persisted_path))
    # should be saved in a dir named after a project
    assert normalized_path == tmpdir.strpath


def test_handles_pipeline_with_non_existing_component(component_builder):
    _config = utilities.base_test_conf("pretrained_embeddings_spacy")
    _config.pipeline.append({"name": "my_made_up_component"})
    with pytest.raises(Exception) as execinfo:
        train(_config, data=DEFAULT_DATA_PATH, component_builder=component_builder)
    assert "Failed to find component" in str(execinfo.value)


@pytest.mark.parametrize("language, pipeline", pipelines_for_tests())
def test_load_and_persist_without_train(language, pipeline, component_builder, tmpdir):
    _config = RasaNLUModelConfig({"pipeline": pipeline, "language": language})
    trainer = Trainer(_config, component_builder)
    persistor = create_persistor(_config)
    persisted_path = trainer.persist(tmpdir.strpath, persistor)
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
    persisted_path = trainer.persist(tmpdir.strpath, persistor)
    loaded = Interpreter.load(persisted_path, component_builder)
    assert loaded.pipeline
    assert loaded.parse("hello") is not None
    assert loaded.parse("Hello today is Monday, again!") is not None
