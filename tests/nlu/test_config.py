import os
from typing import Text
from unittest.mock import Mock

import pytest
from _pytest.monkeypatch import MonkeyPatch

from rasa.shared.exceptions import YamlSyntaxException
from rasa.shared.importers import autoconfig
from rasa.shared.importers.rasa import RasaFileImporter
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu import config
from rasa.nlu.components import ComponentBuilder
from rasa.nlu.constants import COMPONENT_INDEX
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

    assert _config.for_component(idx_tokenizer) == {
        "name": "SpacyTokenizer",
        COMPONENT_INDEX: idx_tokenizer,
    }
    assert _config.for_component(idx_classifier) == {
        "name": "DIETClassifier",
        "epochs": 10,
        COMPONENT_INDEX: idx_classifier,
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
    imported_config = importer.get_config()

    loaded_config = config.load(imported_config)

    assert len(loaded_config.component_names) > 1
    assert loaded_config.language == imported_config["language"]
