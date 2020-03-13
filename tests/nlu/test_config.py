import json
import tempfile
import os
from typing import Text

import pytest

import rasa.utils.io as io_utils
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu import config
from rasa.nlu.components import ComponentBuilder
from rasa.nlu.registry import registered_pipeline_templates
from rasa.nlu.model import Trainer
from tests.nlu.utilities import write_file_config


def test_blank_config(blank_config):
    file_config = {}
    f = write_file_config(file_config)
    final_config = config.load(f.name)

    assert final_config.as_dict() == blank_config.as_dict()


def test_invalid_config_json():
    file_config = """pipeline: [pretrained_embeddings_spacy"""  # invalid yaml

    with tempfile.NamedTemporaryFile("w+", suffix="_tmp_config_file.json") as f:
        f.write(file_config)
        f.flush()

        with pytest.raises(config.InvalidConfigError):
            config.load(f.name)


def test_invalid_pipeline_template():
    args = {"pipeline": "my_made_up_name"}
    f = write_file_config(args)

    with pytest.raises(config.InvalidConfigError) as execinfo:
        config.load(f.name)
    assert "unknown pipeline template" in str(execinfo.value)


def test_invalid_many_tokenizers_in_config():
    nlu_config = {
        "pipeline": [{"name": "WhitespaceTokenizer"}, {"name": "SpacyTokenizer"}]
    }

    with pytest.raises(config.InvalidConfigError) as execinfo:
        Trainer(config.RasaNLUModelConfig(nlu_config))
    assert "More then one tokenizer is used" in str(execinfo.value)


@pytest.mark.parametrize(
    "_config",
    [
        {"pipeline": [{"name": "WhitespaceTokenizer"}, {"name": "SpacyFeaturizer"}]},
        {"pipeline": [{"name": "WhitespaceTokenizer"}, {"name": "ConveRTFeaturizer"}]},
        {
            "pipeline": [
                {"name": "ConveRTTokenizer"},
                {"name": "LanguageModelFeaturizer"},
            ]
        },
    ],
)
def test_missing_required_component(_config):
    with pytest.raises(config.InvalidConfigError) as execinfo:
        Trainer(config.RasaNLUModelConfig(_config))
    assert "Add required components to the pipeline" in str(execinfo.value)


@pytest.mark.parametrize(
    "pipeline_config", [{"pipeline": [{"name": "CountVectorsFeaturizer"}]}]
)
def test_missing_property(pipeline_config):
    with pytest.raises(config.InvalidConfigError) as execinfo:
        Trainer(config.RasaNLUModelConfig(pipeline_config))
    assert "Add required components to the pipeline" in str(execinfo.value)


@pytest.mark.parametrize(
    "pipeline_template", list(registered_pipeline_templates.keys())
)
def test_pipeline_registry_lookup(pipeline_template: Text):
    args = {"pipeline": pipeline_template}
    f = write_file_config(args)

    final_config = config.load(f.name)
    components = [c for c in final_config.pipeline]

    assert json.dumps(components, sort_keys=True) == json.dumps(
        registered_pipeline_templates[pipeline_template], sort_keys=True
    )


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
def test_train_docker_and_docs_configs(config_file: Text):
    content = io_utils.read_yaml_file(config_file)

    loaded_config = config.load(config_file)

    assert len(loaded_config.component_names) > 1
    assert loaded_config.language == content["language"]
