import os
import pytest

from rasa.nlu import registry, train
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.model import Interpreter, Trainer
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.utils.tensorflow.constants import EPOCHS
from tests.nlu.conftest import DEFAULT_DATA_PATH
from typing import Any, Dict, List, Tuple, Text, Union

COMPONENTS_TEST_PARAMS = {
    "DIETClassifier": {EPOCHS: 1},
    "ResponseSelector": {EPOCHS: 1},
    "HFTransformersNLP": {"model_name": "bert", "model_weights": "bert-base-uncased"},
    "LanguageModelFeaturizer": {
        "model_name": "bert",
        "model_weights": "bert-base-uncased",
    },
}


def get_test_params_for_component(component: Text) -> Dict[Text, Union[Text, int]]:
    return (
        COMPONENTS_TEST_PARAMS[component] if component in COMPONENTS_TEST_PARAMS else {}
    )


def as_pipeline(*components):
    return [{**{"name": c}, **get_test_params_for_component(c)} for c in components]


def pipelines_for_tests() -> List[Tuple[Text, List[Dict[Text, Any]]]]:
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
                "DucklingEntityExtractor",
                "DIETClassifier",
                "ResponseSelector",
                "EntitySynonymMapper",
            ),
        ),
        (
            "en",
            as_pipeline(
                "SpacyNLP",
                "SpacyTokenizer",
                "SpacyFeaturizer",
                "SpacyEntityExtractor",
                "SklearnIntentClassifier",
            ),
        ),
        (
            "en",
            as_pipeline(
                "HFTransformersNLP",
                "LanguageModelTokenizer",
                "LanguageModelFeaturizer",
                "DIETClassifier",
            ),
        ),
        ("fallback", as_pipeline("KeywordIntentClassifier", "FallbackClassifier")),
    ]


def pipelines_for_non_windows_tests() -> List[Tuple[Text, List[Dict[Text, Any]]]]:
    # these templates really are just for testing

    # because some of the components are not available on Windows, we specify pipelines
    # containing them separately

    # first is language followed by list of components
    return [
        (
            "en",
            as_pipeline(
                "SpacyNLP", "SpacyTokenizer", "SpacyFeaturizer", "DIETClassifier"
            ),
        ),
        (
            "en",
            as_pipeline(
                "MitieNLP",
                "MitieTokenizer",
                "MitieFeaturizer",
                "MitieIntentClassifier",
                "RegexEntityExtractor",
            ),
        ),
        (
            "zh",
            as_pipeline(
                "MitieNLP", "JiebaTokenizer", "MitieFeaturizer", "MitieEntityExtractor"
            ),
        ),
    ]


def test_all_components_are_in_at_least_one_test_pipeline():
    """There is a template that includes all components to
    test the train-persist-load-use cycle. Ensures that
    really all components are in there.
    """
    all_pipelines = pipelines_for_tests() + pipelines_for_non_windows_tests()
    all_components = [c["name"] for _, p in all_pipelines for c in p]

    for cls in registry.component_classes:
        if "convert" in cls.name.lower():
            # TODO
            #   skip ConveRTTokenizer and ConveRTFeaturizer as the ConveRT model is not publicly available anymore
            #   (see https://github.com/RasaHQ/rasa/issues/6806)
            continue
        assert (
            cls.name in all_components
        ), "`all_components` template is missing component."


@pytest.mark.timeout(600)
@pytest.mark.parametrize("language, pipeline", pipelines_for_tests())
async def test_train_persist_load_parse(language, pipeline, component_builder, tmpdir):
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
    assert loaded.parse("Rasa is great!") is not None


@pytest.mark.timeout(600)
@pytest.mark.parametrize("language, pipeline", pipelines_for_non_windows_tests())
@pytest.mark.skip_on_windows
async def test_train_persist_load_parse_non_windows(
    language, pipeline, component_builder, tmpdir
):
    await test_train_persist_load_parse(language, pipeline, component_builder, tmpdir)


@pytest.mark.parametrize("language, pipeline", pipelines_for_tests())
def test_train_model_without_data(language, pipeline, component_builder, tmpdir):
    _config = RasaNLUModelConfig({"pipeline": pipeline, "language": language})

    trainer = Trainer(_config, component_builder)
    trainer.train(TrainingData())
    persisted_path = trainer.persist(tmpdir.strpath)

    loaded = Interpreter.load(persisted_path, component_builder)

    assert loaded.pipeline
    assert loaded.parse("Rasa is great!") is not None


@pytest.mark.timeout(600)
@pytest.mark.parametrize("language, pipeline", pipelines_for_non_windows_tests())
@pytest.mark.skip_on_windows
def test_train_model_without_data_non_windows(
    language, pipeline, component_builder, tmpdir
):
    test_train_model_without_data(language, pipeline, component_builder, tmpdir)


@pytest.mark.timeout(600)
@pytest.mark.parametrize("language, pipeline", pipelines_for_tests())
def test_load_and_persist_without_train(language, pipeline, component_builder, tmpdir):
    _config = RasaNLUModelConfig({"pipeline": pipeline, "language": language})

    trainer = Trainer(_config, component_builder)
    persisted_path = trainer.persist(tmpdir.strpath)

    loaded = Interpreter.load(persisted_path, component_builder)

    assert loaded.pipeline
    assert loaded.parse("Rasa is great!") is not None


@pytest.mark.timeout(600)
@pytest.mark.parametrize("language, pipeline", pipelines_for_non_windows_tests())
@pytest.mark.skip_on_windows
def test_load_and_persist_without_train_non_windows(
    language, pipeline, component_builder, tmpdir
):
    test_load_and_persist_without_train(language, pipeline, component_builder, tmpdir)


async def test_train_model_empty_pipeline(component_builder):
    _config = RasaNLUModelConfig({"pipeline": None, "language": "en"})

    with pytest.raises(ValueError):
        await train(
            _config, data=DEFAULT_DATA_PATH, component_builder=component_builder
        )


async def test_train_named_model(component_builder, tmpdir):
    _config = RasaNLUModelConfig(
        {"pipeline": [{"name": "KeywordIntentClassifier"}], "language": "en"}
    )

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


async def test_train_model_training_data_persisted(component_builder, tmpdir):
    _config = RasaNLUModelConfig(
        {"pipeline": [{"name": "KeywordIntentClassifier"}], "language": "en"}
    )

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


async def test_train_model_no_training_data_persisted(component_builder, tmpdir):
    _config = RasaNLUModelConfig(
        {"pipeline": [{"name": "KeywordIntentClassifier"}], "language": "en"}
    )

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
