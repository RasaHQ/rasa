import logging

import pytest
from rasa.nlu import data_router, config
from rasa.nlu.components import ComponentBuilder
from rasa.nlu.model import Trainer
from rasa.nlu import training_data
from rasa.nlu.config import RasaNLUModelConfig

logging.basicConfig(level="DEBUG")

CONFIG_DEFAULTS_PATH = "sample_configs/config_defaults.yml"

DEFAULT_DATA_PATH = "data/examples/rasa/demo-rasa.json"

TEST_MODEL_PATH = "test_models/test_model_pretrained_embeddings"

TEST_PROJECTS_PATH = "test_projects"
# see `rasa.nlu.data_router` for details. avoids deadlock in
# `deferred_from_future` function during tests
data_router.DEFERRED_RUN_IN_REACTOR_THREAD = False


@pytest.fixture(scope="session")
def component_builder():
    return ComponentBuilder()


@pytest.fixture(scope="session")
def spacy_nlp(component_builder, default_config):
    spacy_nlp_config = {"name": "SpacyNLP"}
    return component_builder.create_component(spacy_nlp_config, default_config).nlp


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


@pytest.fixture(scope="session")
def trained_nlu_model():
    cfg = RasaNLUModelConfig({"pipeline": "keyword"})
    trainer = Trainer(cfg)
    td = training_data.load_data(DEFAULT_DATA_PATH)

    trainer.train(td)
    model_path = trainer.persist("test_models", project_name="test_model_keyword")

    return model_path
