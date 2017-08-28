from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

import pytest

from rasa_nlu.components import ComponentBuilder
from rasa_nlu.config import RasaNLUConfig

pytest_plugins = str("pytest_twisted")

logging.basicConfig(level="DEBUG")

CONFIG_DEFAULTS_PATH = "sample_configs/config_defaults.json"


@pytest.fixture(scope="session")
def component_builder():
    return ComponentBuilder()


@pytest.fixture(scope="session")
def spacy_nlp(component_builder, default_config):
    return component_builder.create_component("nlp_spacy", default_config).nlp


@pytest.fixture(scope="session")
def mitie_feature_extractor(component_builder, default_config):
    return component_builder.create_component("nlp_mitie", default_config).extractor


@pytest.fixture(scope="session")
def default_config():
    return RasaNLUConfig(CONFIG_DEFAULTS_PATH)
