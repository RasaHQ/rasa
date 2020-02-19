import os

import pytest

from rasa.nlu import registry, train
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.model import Interpreter, Trainer
from rasa.nlu.train import create_persistor
from rasa.nlu.training_data import TrainingData
from rasa.utils.tensorflow.constants import EPOCHS
from tests.nlu import utilities
from tests.nlu.conftest import DEFAULT_DATA_PATH


def as_pipeline(*components):
    return [{"name": c, EPOCHS: 2} for c in components]


def pipelines_for_tests():
    # these templates really are just for testing
    # every component should be in here so train-persist-load-use cycle can be
    # tested they still need to be in a useful order - hence we can not simply
    # generate this automatically.

    # Create separate test pipelines for dense featurizers
    # because they can't co-exist in the same pipeline together,
    # as their tokenizers break the incoming message into different number of tokens.

    # first is language followed by list of components
    return [
        ("en", as_pipeline("KeywordIntentClassifier")),
        (
            "en",
            as_pipeline(
                "WhitespaceTokenizer",
                "RegexFeaturizer",
                "LexicalSyntacticFeaturizer",
                "CountVectorsFeaturizer",
                "CRFEntityExtractor",
                "DucklingHTTPExtractor",
                "DIETClassifier",
                "EmbeddingIntentClassifier",
                "ResponseSelector",
                "DIETSelector",
                "EntitySynonymMapper",
            ),
        ),
        (
            "en",
            as_pipeline(
                "SpacyNLP",
                "SpacyTokenizer",
                "SpacyFeaturizer",
                "RegexFeaturizer",
                "LexicalSyntacticFeaturizer",
                "CountVectorsFeaturizer",
                "CRFEntityExtractor",
                "DucklingHTTPExtractor",
                "SpacyEntityExtractor",
                "SklearnIntentClassifier",
                "DIETClassifier",
                "ResponseSelector",
                "DIETSelector",
                "EntitySynonymMapper",
            ),
        ),
        (
            "en",
            as_pipeline(
                "HFTransformersNLP",
                "LanguageModelTokenizer",
                "LanguageModelFeaturizer",
                "RegexFeaturizer",
                "LexicalSyntacticFeaturizer",
                "CountVectorsFeaturizer",
                "CRFEntityExtractor",
                "DucklingHTTPExtractor",
                "DIETClassifier",
                "ResponseSelector",
                "DIETSelector",
                "EntitySynonymMapper",
            ),
        ),
        (
            "en",
            as_pipeline(
                "ConveRTTokenizer",
                "ConveRTFeaturizer",
                "RegexFeaturizer",
                "LexicalSyntacticFeaturizer",
                "CountVectorsFeaturizer",
                "CRFEntityExtractor",
                "DucklingHTTPExtractor",
                "DIETClassifier",
                "ResponseSelector",
                "DIETSelector",
                "EntitySynonymMapper",
            ),
        ),
        (
            "en",
            as_pipeline(
                "MitieNLP",
                "MitieTokenizer",
                "MitieFeaturizer",
                "RegexFeaturizer",
                "CountVectorsFeaturizer",
                "MitieEntityExtractor",
                "DucklingHTTPExtractor",
                "MitieIntentClassifier",
                "DIETClassifier",
                "ResponseSelector",
                "DIETSelector",
                "EntitySynonymMapper",
            ),
        ),
        (
            "zh",
            as_pipeline(
                "MitieNLP",
                "JiebaTokenizer",
                "MitieFeaturizer",
                "RegexFeaturizer",
                "CountVectorsFeaturizer",
                "MitieEntityExtractor",
                "MitieIntentClassifier",
                "DIETClassifier",
                "ResponseSelector",
                "DIETSelector",
                "EntitySynonymMapper",
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


@pytest.mark.parametrize(
    "pipeline_template", list(registry.registered_pipeline_templates.keys())
)
async def test_train_model(pipeline_template, component_builder, tmpdir):
    _config = utilities.base_test_conf(pipeline_template)
    (trained, _, persisted_path) = await train(
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


async def test_random_seed(component_builder, tmpdir, supervised_embeddings_config):
    """test if train result is the same for two runs of tf embedding"""

    # set fixed random seed
    idx = supervised_embeddings_config.component_names.index(
        "EmbeddingIntentClassifier"
    )
    supervised_embeddings_config.set_component_attr(idx, random_seed=1)
    idx = supervised_embeddings_config.component_names.index("CRFEntityExtractor")
    supervised_embeddings_config.set_component_attr(idx, random_seed=1)

    # first run
    (trained_a, _, persisted_path_a) = await train(
        supervised_embeddings_config,
        path=tmpdir.strpath + "_a",
        data=DEFAULT_DATA_PATH,
        component_builder=component_builder,
    )
    # second run
    (trained_b, _, persisted_path_b) = await train(
        supervised_embeddings_config,
        path=tmpdir.strpath + "_b",
        data=DEFAULT_DATA_PATH,
        component_builder=component_builder,
    )
    loaded_a = Interpreter.load(persisted_path_a, component_builder)
    loaded_b = Interpreter.load(persisted_path_b, component_builder)
    result_a = loaded_a.parse("hello")["intent"]["confidence"]
    result_b = loaded_b.parse("hello")["intent"]["confidence"]
    assert result_a == result_b


@pytest.mark.parametrize("language, pipeline", pipelines_for_tests())
async def test_train_model_on_test_pipelines(
    language, pipeline, component_builder, tmpdir
):
    _config = RasaNLUModelConfig({"pipeline": pipeline, "language": language})
    (trained, _, persisted_path) = await train(
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


@pytest.mark.parametrize("language, pipeline", pipelines_for_tests())
async def test_train_model_no_events(language, pipeline, component_builder, tmpdir):
    _config = RasaNLUModelConfig({"pipeline": pipeline, "language": language})
    (trained, _, persisted_path) = await train(
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


async def test_train_model_empty_pipeline(component_builder):
    # Should return an empty pipeline
    _config = utilities.base_test_conf(pipeline_template=None)
    with pytest.raises(ValueError):
        await train(
            _config, data=DEFAULT_DATA_PATH, component_builder=component_builder
        )


async def test_train_named_model(component_builder, tmpdir):
    _config = utilities.base_test_conf("keyword")
    (trained, _, persisted_path) = await train(
        _config,
        path=tmpdir.strpath,
        data=DEFAULT_DATA_PATH,
        component_builder=component_builder,
    )
    assert trained.pipeline
    normalized_path = os.path.dirname(os.path.normpath(persisted_path))
    # should be saved in a dir named after a project
    assert normalized_path == tmpdir.strpath


async def test_handles_pipeline_with_non_existing_component(
    component_builder, pretrained_embeddings_spacy_config
):
    pretrained_embeddings_spacy_config.pipeline.append({"name": "my_made_up_component"})
    with pytest.raises(Exception) as execinfo:
        await train(
            pretrained_embeddings_spacy_config,
            data=DEFAULT_DATA_PATH,
            component_builder=component_builder,
        )
    assert "Cannot find class" in str(execinfo.value)


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


async def test_train_model_no_training_data_persisted(component_builder, tmpdir):
    _config = utilities.base_test_conf("keyword")
    (trained, _, persisted_path) = await train(
        _config,
        path=tmpdir.strpath,
        data=DEFAULT_DATA_PATH,
        component_builder=component_builder,
        persist_nlu_training_data=False,
    )
    assert trained.pipeline
    loaded = Interpreter.load(persisted_path, component_builder)
    assert loaded.pipeline
    assert loaded.model_metadata.get("training_data") is None


async def test_train_model_training_data_persisted(component_builder, tmpdir):
    _config = utilities.base_test_conf("keyword")
    (trained, _, persisted_path) = await train(
        _config,
        path=tmpdir.strpath,
        data=DEFAULT_DATA_PATH,
        component_builder=component_builder,
        persist_nlu_training_data=True,
    )
    assert trained.pipeline
    loaded = Interpreter.load(persisted_path, component_builder)
    assert loaded.pipeline
    assert loaded.model_metadata.get("training_data") is not None
