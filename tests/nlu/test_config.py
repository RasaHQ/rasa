import os
from typing import Text, List
from unittest.mock import Mock

import pytest
from _pytest.monkeypatch import MonkeyPatch

from rasa.shared.exceptions import InvalidConfigException, YamlSyntaxException
from rasa.shared.importers import autoconfig
from rasa.shared.importers.rasa import RasaFileImporter
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu import config
import rasa.shared.nlu.training_data.loading
from rasa.nlu import components
from rasa.nlu.components import ComponentBuilder
from rasa.shared.nlu.constants import TRAINABLE_EXTRACTORS
from rasa.nlu.model import Trainer
from tests.nlu.utilities import write_file_config


def test_blank_config(blank_config):
    file_config = {}
    f = write_file_config(file_config)
    final_config = config.load(f.name)

    assert final_config.as_dict() == blank_config.as_dict()


def test_invalid_config_json(tmp_path):
    file_config = """pipeline: [pretrained_embeddings_spacy"""  # invalid yaml

    f = tmp_path / "tmp_config_file.json"
    f.write_text(file_config)

    with pytest.raises(YamlSyntaxException):
        config.load(str(f))


def test_invalid_many_tokenizers_in_config():
    nlu_config = {
        "pipeline": [{"name": "WhitespaceTokenizer"}, {"name": "SpacyTokenizer"}]
    }

    with pytest.raises(InvalidConfigException) as execinfo:
        Trainer(config.RasaNLUModelConfig(nlu_config))
    assert "The pipeline configuration contains more than one" in str(execinfo.value)


@pytest.mark.parametrize(
    "_config",
    [
        {"pipeline": [{"name": "WhitespaceTokenizer"}, {"name": "SpacyFeaturizer"}]},
        pytest.param(
            {
                "pipeline": [
                    {"name": "WhitespaceTokenizer"},
                    {"name": "MitieIntentClassifier"},
                ]
            }
        ),
    ],
)
@pytest.mark.skip_on_windows
def test_missing_required_component(_config):
    with pytest.raises(InvalidConfigException) as execinfo:
        Trainer(config.RasaNLUModelConfig(_config))
    assert "The pipeline configuration contains errors" in str(execinfo.value)


@pytest.mark.parametrize(
    "pipeline_config", [{"pipeline": [{"name": "CountVectorsFeaturizer"}]}]
)
def test_missing_property(pipeline_config):
    with pytest.raises(InvalidConfigException) as execinfo:
        Trainer(config.RasaNLUModelConfig(pipeline_config))
    assert "The pipeline configuration contains errors" in str(execinfo.value)


def test_default_config_file():
    final_config = config.RasaNLUModelConfig()
    assert len(final_config) > 1


def test_set_attr_on_component():
    _config = RasaNLUModelConfig(
        {
            "language": "en",
            "pipeline": [
                {"name": "SpacyNLP"},
                {"name": "SpacyTokenizer"},
                {"name": "SpacyFeaturizer"},
                {"name": "DIETClassifier"},
            ],
        }
    )
    idx_classifier = _config.component_names.index("DIETClassifier")
    idx_tokenizer = _config.component_names.index("SpacyTokenizer")

    _config.set_component_attr(idx_classifier, epochs=10)

    assert _config.for_component(idx_tokenizer) == {"name": "SpacyTokenizer"}
    assert _config.for_component(idx_classifier) == {
        "name": "DIETClassifier",
        "epochs": 10,
    }


def test_override_defaults_supervised_embeddings_pipeline():
    builder = ComponentBuilder()

    _config = RasaNLUModelConfig(
        {
            "language": "en",
            "pipeline": [
                {"name": "SpacyNLP"},
                {"name": "SpacyTokenizer"},
                {"name": "SpacyFeaturizer", "pooling": "max"},
                {
                    "name": "DIETClassifier",
                    "epochs": 10,
                    "hidden_layers_sizes": {"text": [256, 128]},
                },
            ],
        }
    )

    idx_featurizer = _config.component_names.index("SpacyFeaturizer")
    idx_classifier = _config.component_names.index("DIETClassifier")

    component1 = builder.create_component(
        _config.for_component(idx_featurizer), _config
    )
    assert component1.component_config["pooling"] == "max"

    component2 = builder.create_component(
        _config.for_component(idx_classifier), _config
    )
    assert component2.component_config["epochs"] == 10
    assert (
        component2.defaults["hidden_layers_sizes"].keys()
        == component2.component_config["hidden_layers_sizes"].keys()
    )


def config_files_in(config_directory: Text):
    return [
        os.path.join(config_directory, f)
        for f in os.listdir(config_directory)
        if os.path.isfile(os.path.join(config_directory, f))
    ]


@pytest.mark.parametrize(
    "config_file",
    config_files_in("data/configs_for_docs") + config_files_in("docker/configs"),
)
async def test_train_docker_and_docs_configs(
    config_file: Text, monkeypatch: MonkeyPatch
):
    monkeypatch.setattr(autoconfig, "_dump_config", Mock())
    importer = RasaFileImporter(config_file=config_file)
    imported_config = await importer.get_config()

    loaded_config = config.load(imported_config)

    assert len(loaded_config.component_names) > 1
    assert loaded_config.language == imported_config["language"]


@pytest.mark.parametrize(
    "config_path, data_path, expected_warning_excerpts",
    [
        (
            "data/test_config/config_supervised_embeddings.yml",
            "data/examples/rasa",
            ["add a 'ResponseSelector'"],
        ),
        (
            "data/test_config/config_spacy_entity_extractor.yml",
            "data/test/duplicate_intents_markdown/demo-rasa-intents-2.md",
            [f"add one of {TRAINABLE_EXTRACTORS}"],
        ),
        (
            "data/test_config/config_crf_no_regex.yml",
            "data/test/duplicate_intents_markdown/demo-rasa-intents-2.md",
            ["training data with regexes", "include a 'RegexFeaturizer'"],
        ),
        (
            "data/test_config/config_crf_no_regex.yml",
            "data/test/lookup_tables/lookup_table.json",
            ["training data consisting of lookup tables", "add a 'RegexFeaturizer'"],
        ),
        (
            "data/test_config/config_spacy_entity_extractor.yml",
            "data/test/lookup_tables/lookup_table.json",
            [
                "add a 'DIETClassifier' or a 'CRFEntityExtractor' with the 'pattern' feature"
            ],
        ),
        (
            "data/test_config/config_crf_no_pattern_feature.yml",
            "data/test/lookup_tables/lookup_table.md",
            "your NLU pipeline's 'CRFEntityExtractor' does not include the 'pattern' feature",
        ),
        (
            "data/test_config/config_crf_no_synonyms.yml",
            "data/test/markdown_single_sections/synonyms_only.md",
            ["add an 'EntitySynonymMapper'"],
        ),
        (
            "data/test_config/config_embedding_intent_response_selector.yml",
            "data/test/demo-rasa-composite-entities.md",
            ["include either 'DIETClassifier' or 'CRFEntityExtractor'"],
        ),
    ],
)
def test_validate_required_components_from_data(
    config_path: Text, data_path: Text, expected_warning_excerpts: List[Text]
):
    loaded_config = config.load(config_path)
    trainer = Trainer(loaded_config)
    training_data = rasa.shared.nlu.training_data.loading.load_data(data_path)
    with pytest.warns(UserWarning) as record:
        components.validate_required_components_from_data(
            trainer.pipeline, training_data
        )
    assert len(record) == 1
    assert all(
        [excerpt in record[0].message.args[0]] for excerpt in expected_warning_excerpts
    )
