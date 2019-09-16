import logging
import os

import pytest

from rasa.nlu import config, train
from rasa.nlu.components import ComponentBuilder

CONFIG_DEFAULTS_PATH = "sample_configs/config_defaults.yml"

NLU_DEFAULT_CONFIG_PATH = "sample_configs/config_pretrained_embeddings_mitie.yml"

DEFAULT_DATA_PATH = "data/examples/rasa/demo-rasa.json"

NLU_MODEL_NAME = "nlu_model.tar.gz"

TEST_MODEL_DIR = "test_models"

NLU_MODEL_PATH = os.path.join(TEST_MODEL_DIR, "nlu")

MOODBOT_MODEL_PATH = "examples/moodbot/models/"


@pytest.fixture(scope="session")
def component_builder():
    return ComponentBuilder()


@pytest.fixture(scope="session")
def spacy_nlp(component_builder, default_config):
    spacy_nlp_config = {"name": "SpacyNLP"}
    return component_builder.create_component(spacy_nlp_config, default_config).nlp


@pytest.fixture(scope="session")
def spacy_nlp_component(component_builder, default_config):
    spacy_nlp_config = {"name": "SpacyNLP"}
    return component_builder.create_component(spacy_nlp_config, default_config)


@pytest.fixture(scope="session")
def ner_crf_pos_feature_config():
    return {
        "features": [
            ["low", "title", "upper", "pos", "pos2"],
            [
                "bias",
                "low",
                "suffix3",
                "suffix2",
                "upper",
                "title",
                "digit",
                "pos",
                "pos2",
                "pattern",
            ],
            ["low", "title", "upper", "pos", "pos2"],
        ]
    }


@pytest.fixture(scope="session")
def mitie_feature_extractor(component_builder, default_config):
    mitie_nlp_config = {"name": "MitieNLP"}
    return component_builder.create_component(
        mitie_nlp_config, default_config
    ).extractor


@pytest.fixture(scope="session")
def default_config():
    return config.load(CONFIG_DEFAULTS_PATH)
