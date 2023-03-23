from pathlib import Path

import pytest
from _pytest.tmpdir import TempPathFactory

from rasa.core.agent import Agent
from rasa.core.policies.policy import Policy
from rasa.engine.storage.local_model_storage import LocalModelStorage
from rasa.shared.nlu.training_data.formats import RasaYAMLReader
from rasa.utils.tensorflow.constants import EPOCHS, RUN_EAGERLY
from typing import Any, Dict, List, Tuple, Text, Union, Optional
import rasa.model_training
import rasa.shared.utils.io
import rasa.engine.recipes.default_components

COMPONENTS_TEST_PARAMS = {
    "DIETClassifier": {EPOCHS: 1, RUN_EAGERLY: True},
    "ResponseSelector": {EPOCHS: 1, RUN_EAGERLY: True},
    "LanguageModelFeaturizer": {
        "model_name": "bert",
        "model_weights": "sentence-transformers/all-MiniLM-L6-v2",
    },
}


def get_test_params_for_component(component: Text) -> Dict[Text, Union[Text, int]]:
    return (
        COMPONENTS_TEST_PARAMS[component] if component in COMPONENTS_TEST_PARAMS else {}
    )


def as_pipeline(*components) -> List[Dict[Text, Dict]]:
    return [
        {"name": c, **get_test_params_for_component(c)} if isinstance(c, str) else c
        for c in components
    ]


def pipelines_for_tests() -> List[Tuple[Text, List[Dict[Text, Any]]]]:
    # these pipelines really are just for testing
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
                {"name": "SpacyNLP", "model": "en_core_web_md"},
                "SpacyTokenizer",
                "SpacyFeaturizer",
                "SpacyEntityExtractor",
                "SklearnIntentClassifier",
            ),
        ),
        (
            "en",
            as_pipeline(
                {"name": "SpacyNLP", "model": "en_core_web_md"},
                "SpacyTokenizer",
                "SpacyFeaturizer",
                "CountVectorsFeaturizer",
                "LogisticRegressionClassifier",
            ),
        ),
        (
            "en",
            as_pipeline(
                "WhitespaceTokenizer", "LanguageModelFeaturizer", "DIETClassifier"
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
                {"name": "SpacyNLP", "model": "en_core_web_md"},
                "SpacyTokenizer",
                "SpacyFeaturizer",
                "DIETClassifier",
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

    all_registered_components = (
        rasa.engine.recipes.default_components.DEFAULT_COMPONENTS
    )
    all_registered_nlu_components = [
        c for c in all_registered_components if not issubclass(c, Policy)
    ]

    for cls in all_registered_nlu_components:
        if "convert" in cls.__name__.lower():
            # TODO
            #   skip ConveRTFeaturizer as the ConveRT model is not
            #   publicly available anymore
            #   (see https://github.com/RasaHQ/rasa/issues/6806)
            continue
        assert (
            cls.__name__ in all_components
        ), "`all_components` template is missing component."


@pytest.mark.timeout(600, func_only=True)
@pytest.mark.parametrize("language, pipeline", pipelines_for_tests())
async def test_train_persist_load_parse(
    language: Optional[Text],
    pipeline: List[Dict],
    tmp_path: Path,
    nlu_as_json_path: Text,
):
    config_file = tmp_path / "config.yml"
    rasa.shared.utils.io.dump_obj_as_json_to_file(
        config_file,
        {
            "pipeline": pipeline,
            "language": language,
            "assistant_id": "placeholder_default",
        },
    )

    persisted_path = rasa.model_training.train_nlu(
        str(config_file), nlu_as_json_path, output=str(tmp_path)
    )

    assert Path(persisted_path).is_file()

    agent = Agent.load(persisted_path)
    assert agent.processor
    assert agent.is_ready()
    assert await agent.parse_message("Rasa is great!") is not None


@pytest.mark.timeout(600, func_only=True)
@pytest.mark.parametrize("language, pipeline", pipelines_for_non_windows_tests())
@pytest.mark.skip_on_windows
def test_train_persist_load_parse_non_windows(
    language, pipeline, tmp_path, nlu_as_json_path: Text
):
    test_train_persist_load_parse(language, pipeline, tmp_path, nlu_as_json_path)


def test_train_model_empty_pipeline(nlu_as_json_path: Text, tmp_path: Path):
    config_file = tmp_path / "config.yml"
    rasa.shared.utils.io.dump_obj_as_json_to_file(
        config_file, {"pipeline": [], "assistant_id": "placeholder_default"}
    )

    with pytest.raises(ValueError):
        rasa.model_training.train_nlu(
            str(config_file), nlu_as_json_path, output=str(tmp_path)
        )


def test_handles_pipeline_with_non_existing_component(
    tmp_path: Path, pretrained_embeddings_spacy_config: Dict, nlu_as_json_path: Text
):
    pretrained_embeddings_spacy_config["pipeline"].append(
        {"name": "my_made_up_component"}
    )

    config_file = tmp_path / "config.yml"
    rasa.shared.utils.io.dump_obj_as_json_to_file(
        config_file, pretrained_embeddings_spacy_config
    )

    with pytest.raises(
        Exception, match="Can't load class for name 'my_made_up_component'"
    ):
        rasa.model_training.train_nlu(
            str(config_file), nlu_as_json_path, output=str(tmp_path)
        )


def test_train_model_training_data_persisted(
    tmp_path: Path, nlu_as_json_path: Text, tmp_path_factory: TempPathFactory
):
    config_file = tmp_path / "config.yml"
    rasa.shared.utils.io.dump_obj_as_json_to_file(
        config_file,
        {
            "pipeline": [{"name": "KeywordIntentClassifier"}],
            "language": "en",
            "assistant_id": "placeholder_default",
        },
    )

    persisted_path = rasa.model_training.train_nlu(
        str(config_file),
        nlu_as_json_path,
        output=str(tmp_path),
        persist_nlu_training_data=True,
    )

    assert Path(persisted_path).is_file()

    model_dir = tmp_path_factory.mktemp("loaded")
    storage, _ = LocalModelStorage.from_model_archive(model_dir, Path(persisted_path))

    nlu_data_dir = model_dir / "nlu_training_data_provider"

    assert nlu_data_dir.is_dir()

    assert not RasaYAMLReader().read(nlu_data_dir / "training_data.yml").is_empty()


def test_train_model_no_training_data_persisted(
    tmp_path: Path, nlu_as_json_path: Text, tmp_path_factory: TempPathFactory
):
    config_file = tmp_path / "config.yml"
    rasa.shared.utils.io.dump_obj_as_json_to_file(
        config_file,
        {
            "pipeline": [{"name": "KeywordIntentClassifier"}],
            "language": "en",
            "assistant_id": "placeholder_default",
        },
    )

    persisted_path = rasa.model_training.train_nlu(
        str(config_file),
        nlu_as_json_path,
        output=str(tmp_path),
        persist_nlu_training_data=False,
    )

    assert Path(persisted_path).is_file()

    model_dir = tmp_path_factory.mktemp("loaded")
    storage, _ = LocalModelStorage.from_model_archive(model_dir, Path(persisted_path))

    nlu_data_dir = model_dir / "nlu_training_data_provider"

    assert not nlu_data_dir.is_dir()
