from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

import pytest

from rasa_nlu import data_router, config
from rasa_nlu.components import ComponentBuilder

logging.basicConfig(level="DEBUG")

CONFIG_DEFAULTS_PATH = "sample_configs/config_defaults.yml"

DEFAULT_DATA_PATH = "data/examples/rasa/demo-rasa.json"

# see `rasa_nlu.data_router` for details. avoids deadlock in
# `deferred_from_future` function during tests
data_router.DEFERRED_RUN_IN_REACTOR_THREAD = False


@pytest.fixture(scope="session")
def component_builder():
    return ComponentBuilder()


@pytest.fixture(scope="session")
def spacy_nlp(component_builder, default_config):
    return component_builder.create_component("nlp_spacy", default_config).nlp


@pytest.fixture(scope="session")
def ner_crf_pos_feature_config():
    return {
        "features": [
            ["low", "title", "upper", "pos", "pos2"],
            ["bias", "low", "suffix3", "suffix2", "upper",
             "title", "digit", "pos", "pos2", "pattern"],
            ["low", "title", "upper", "pos", "pos2"]]
    }


@pytest.fixture(scope="session")
def mitie_feature_extractor(component_builder, default_config):
    return component_builder.create_component("nlp_mitie", default_config).extractor


@pytest.fixture(scope="session")
def default_config():
    return config.load(CONFIG_DEFAULTS_PATH)
